import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";

const CHROMA_API_KEY = process.env.CHROMA_API_KEY;
const CHROMA_TENANT = process.env.CHROMA_TENANT;
const CHROMA_DATABASE = process.env.CHROMA_DATABASE;

const COLLECTION_NAME = "FullProjectCollection";
const MAX_CHUNK_LENGTH = 1000;
const ALLOWED_EXTENSIONS = ['.js', '.ts', '.go', '.groovy', '.java', '.html', '.css', '.md'];
const EXCLUDED_FOLDERS = ['node_modules', 'target', 'dist', '.git'];

function makeId(file, chunkId) {
    const hash = crypto.createHash('sha256').update(file).digest('hex').slice(0, 12);
    return `${hash}_p${chunkId}`;
}

function splitText(text, maxLength) {
    const chunks = [];
    for (let i = 0; i < text.length; i += maxLength) {
        chunks.push(text.slice(i, i + maxLength));
    }
    return chunks;
}

async function getFiles(dir) {
    let results = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
        if (EXCLUDED_FOLDERS.includes(entry.name)) continue;
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            results = results.concat(await getFiles(fullPath));
        } else if (ALLOWED_EXTENSIONS.includes(path.extname(entry.name))) {
            results.push(fullPath);
        }
    }
    return results;
}

async function main() {
    console.log("🚀 Loading embedding model...");
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        quantized: true,
        device: "cpu",
        pooling: "mean",
        normalize: true
    });
    console.log("✅ Model loaded");

    const projectRoot = path.resolve('./');
    const files = await getFiles(projectRoot);
    console.log(`Found ${files.length} files to vectorize`);

    const embeddingsMap = {};

    for (const file of files) {
        try {
            const text = await fs.readFile(file, 'utf-8');
            const chunks = splitText(text, MAX_CHUNK_LENGTH);
            embeddingsMap[file] = [];

            for (let i = 0; i < chunks.length; i++) {
                const embeddingOutput = await embedder(chunks[i]);
                let embedding = Array.from(embeddingOutput.data || []);
                embeddingsMap[file].push({ chunkId: i, embedding });
                console.log(`Chunk ${i} of ${file} embedded, length: ${embedding.length}`);
            }
        } catch (err) {
            console.error(`❌ Failed reading/embedding ${file}:`, err.message);
        }
    }

    // Upload to Chroma
    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        console.log("🔌 Connecting to Chroma Cloud...");
        const client = new CloudClient({ apiKey: CHROMA_API_KEY, tenant: CHROMA_TENANT, database: CHROMA_DATABASE });
        let collection;

        try {
            collection = await client.getCollection({ name: COLLECTION_NAME });
            console.log(`✅ Collection exists: ${collection.name}`);
        } catch {
            collection = await client.createCollection({ name: COLLECTION_NAME });
            console.log(`✅ Collection created: ${collection.name}`);
        }

        for (const [file, chunks] of Object.entries(embeddingsMap)) {
            for (const { chunkId, embedding } of chunks) {
                try {
                    await collection.add({
                        ids: [makeId(file, chunkId)],
                        embeddings: [embedding],
                        metadatas: [{ file, chunkId }],
                    });
                    console.log(`✅ Pushed ${file} chunk ${chunkId}`);
                } catch (err) {
                    console.error(`❌ Failed to push ${file} chunk ${chunkId}:`, err.message);
                }
            }
        }
    } else {
        console.log("Chroma credentials missing, skipping upload");
    }

    console.log("Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));
