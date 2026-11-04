import fsp from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";

function u16(buffer, offset) {
  return buffer.readUInt16LE(offset);
}

function u32(buffer, offset) {
  return buffer.readUInt32LE(offset);
}

function u64(buffer, offset) {
  return Number(buffer.readBigUInt64LE(offset));
}

async function exists(filePath) {
  try {
    await fsp.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function findLastSignature(buffer, signature) {
  for (let i = buffer.length - 4; i >= 0; i -= 1) {
    if (buffer.readUInt32LE(i) === signature) return i;
  }
  return -1;
}

function parseZip64CentralExtra(extra, compressed32, uncompressed32, offset32) {
  let compressed = compressed32;
  let uncompressed = uncompressed32;
  let localOffset = offset32;
  let cursor = 0;

  while (cursor + 4 <= extra.length) {
    const id = u16(extra, cursor);
    const length = u16(extra, cursor + 2);
    cursor += 4;
    const end = cursor + length;

    if (id === 0x0001) {
      let q = cursor;
      if (uncompressed32 === 0xffffffff && q + 8 <= end) {
        uncompressed = u64(extra, q);
        q += 8;
      }
      if (compressed32 === 0xffffffff && q + 8 <= end) {
        compressed = u64(extra, q);
        q += 8;
      }
      if (offset32 === 0xffffffff && q + 8 <= end) {
        localOffset = u64(extra, q);
      }
    }
    cursor = end;
  }

  return { compressed, uncompressed, localOffset };
}

async function readAt(fileHandle, size, position) {
  const buffer = Buffer.alloc(size);
  const result = await fileHandle.read(buffer, 0, size, position);
  if (result.bytesRead !== size) {
    throw new Error(`Short read at ${position}`);
  }
  return buffer;
}

async function getCentralDirectory(zipPath) {
  const stat = await fsp.stat(zipPath);
  const tailSize = Math.min(stat.size, 1024 * 1024);
  const fileHandle = await fsp.open(zipPath, "r");

  try {
    const tail = await readAt(fileHandle, tailSize, stat.size - tailSize);
    const eocdRelative = findLastSignature(tail, 0x06054b50);
    if (eocdRelative < 0) throw new Error(`EOCD not found: ${zipPath}`);

    const eocdAbsolute = stat.size - tailSize + eocdRelative;
    let totalEntries = u16(tail, eocdRelative + 10);
    let centralDirectorySize = u32(tail, eocdRelative + 12);
    let centralDirectoryOffset = u32(tail, eocdRelative + 16);

    if (
      totalEntries === 0xffff ||
      centralDirectorySize === 0xffffffff ||
      centralDirectoryOffset === 0xffffffff
    ) {
      const locator = await readAt(fileHandle, 20, eocdAbsolute - 20);
      if (u32(locator, 0) !== 0x07064b50) {
        throw new Error(`Zip64 locator not found: ${zipPath}`);
      }
      const zip64Offset = u64(locator, 8);
      const zip64 = await readAt(fileHandle, 56, zip64Offset);
      if (u32(zip64, 0) !== 0x06064b50) {
        throw new Error(`Zip64 EOCD not found: ${zipPath}`);
      }
      totalEntries = u64(zip64, 32);
      centralDirectorySize = u64(zip64, 40);
      centralDirectoryOffset = u64(zip64, 48);
    }

    const centralDirectory = await readAt(
      fileHandle,
      centralDirectorySize,
      centralDirectoryOffset,
    );

    const entries = [];
    let cursor = 0;
    for (let i = 0; i < totalEntries && cursor < centralDirectory.length; i += 1) {
      if (u32(centralDirectory, cursor) !== 0x02014b50) {
        throw new Error(`Bad central directory header at ${cursor}`);
      }

      const flags = u16(centralDirectory, cursor + 8);
      const method = u16(centralDirectory, cursor + 10);
      const compressed32 = u32(centralDirectory, cursor + 20);
      const uncompressed32 = u32(centralDirectory, cursor + 24);
      const nameLength = u16(centralDirectory, cursor + 28);
      const extraLength = u16(centralDirectory, cursor + 30);
      const commentLength = u16(centralDirectory, cursor + 32);
      const offset32 = u32(centralDirectory, cursor + 42);
      const name = centralDirectory.toString("utf8", cursor + 46, cursor + 46 + nameLength);
      const extra = centralDirectory.subarray(
        cursor + 46 + nameLength,
        cursor + 46 + nameLength + extraLength,
      );
      const parsed = parseZip64CentralExtra(extra, compressed32, uncompressed32, offset32);

      entries.push({
        name,
        flags,
        method,
        compressed: parsed.compressed,
        uncompressed: parsed.uncompressed,
        localOffset: parsed.localOffset,
      });

      cursor += 46 + nameLength + extraLength + commentLength;
    }

    return entries;
  } finally {
    await fileHandle.close();
  }
}

function safeDestination(outputDir, name) {
  const relative = name.replace(/\\/g, "/").replace(/^\/+/, "");
  if (relative.split("/").includes("..")) {
    throw new Error(`Unsafe zip path: ${name}`);
  }
  return path.join(outputDir, relative);
}

async function dataStartForEntry(zipPath, entry) {
  const fileHandle = await fsp.open(zipPath, "r");
  try {
    const header = await readAt(fileHandle, 30, entry.localOffset);
    if (u32(header, 0) !== 0x04034b50) {
      throw new Error(`Bad local header for ${entry.name}`);
    }
    return entry.localOffset + 30 + u16(header, 26) + u16(header, 28);
  } finally {
    await fileHandle.close();
  }
}

async function extractEntry(zipPath, entry, outputDir) {
  const destination = safeDestination(outputDir, entry.name);

  if (entry.name.endsWith("/")) {
    await fsp.mkdir(destination, { recursive: true });
    return { type: "dir", skipped: false, bytes: 0 };
  }

  if (await exists(destination)) {
    const stat = await fsp.stat(destination);
    if (stat.size > 0) {
      return { type: "file", skipped: true, bytes: 0 };
    }
  }

  await fsp.mkdir(path.dirname(destination), { recursive: true });
  const start = await dataStartForEntry(zipPath, entry);
  const fileHandle = await fsp.open(zipPath, "r");
  let compressed;
  try {
    compressed = Buffer.alloc(entry.compressed);
    await fileHandle.read(compressed, 0, entry.compressed, start);
  } finally {
    await fileHandle.close();
  }

  let output;

  if (entry.method === 0) {
    output = compressed;
  } else if (entry.method === 8) {
    output = zlib.inflateRawSync(compressed);
  } else {
    throw new Error(`Unsupported compression method ${entry.method}: ${entry.name}`);
  }

  await fsp.writeFile(destination, output);

  return { type: "file", skipped: false, bytes: entry.uncompressed };
}

async function main() {
  const [zipPath, outputDir] = process.argv.slice(2);
  if (!zipPath || !outputDir) {
    console.error("Usage: node scripts/node_extract_zip.mjs <zipPath> <outputDir>");
    process.exit(2);
  }

  await fsp.mkdir(outputDir, { recursive: true });
  const entries = await getCentralDirectory(zipPath);
  let files = 0;
  let dirs = 0;
  let skipped = 0;
  let bytes = 0;

  for (const entry of entries) {
    let result;
    try {
      result = await extractEntry(zipPath, entry, outputDir);
    } catch (error) {
      error.message = `${error.message} while extracting ${entry.name}`;
      throw error;
    }
    if (result.type === "dir") dirs += 1;
    if (result.type === "file") files += 1;
    if (result.skipped) skipped += 1;
    bytes += result.bytes;
    if ((files + dirs) % 500 === 0) {
      console.log(`[progress] entries=${files + dirs}/${entries.length} skipped=${skipped}`);
    }
  }

  console.log(
    JSON.stringify({
      zipPath,
      outputDir,
      entries: entries.length,
      files,
      dirs,
      skipped,
      bytes,
    }),
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
