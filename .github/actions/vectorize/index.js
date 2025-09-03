import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";

// ====== CONFIG ======
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = "FullProjectCollection";
const MAX_CHUNK_LENGTH = 1000;
const MAX_EMBED_DIM = 3072;
const ALLOWED_EXTENSIONS = ['.js', '.ts', '.go', '.groovy', '.java', '.html', '.css', '.md'];
const EXCLUDED_FOLDERS = ['node_modules', 'target', 'dist', '.git'];

// ====== UTILS ======
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

function resizeEmbedding(embedding, maxDim = MAX_EMBED_DIM) {
    if (embedding.length <= maxDim) return embedding;
    const factor = embedding.length / maxDim;
    const resized = [];
    for (let i = 0; i < maxDim; i++) {
        resized.push(embedding[Math.floor(i * factor)]);
    }
    return resized;
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

// ====== MAIN ======
async function main() {
    try {
        console.log("🚀 Загружаем локальную модель эмбеддингов...");
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true,
            revision: 'main',
            device: 'cpu',
            pooling: 'mean',
            normalize: true
        });
        console.log("✅ Модель загружена");

        const projectRoot = path.resolve('./');
        const filesToVectorize = await getFiles(projectRoot);
        console.log(`📂 Найдено файлов для векторизации: ${filesToVectorize.length}`);

        const embeddingsMap = {};

        for (const file of filesToVectorize) {
            try {
                const text = await fs.readFile(file, 'utf-8');
                const chunks = splitText(text, MAX_CHUNK_LENGTH);
                embeddingsMap[file] = [];

                for (let i = 0; i < chunks.length; i++) {
                    const chunk = chunks[i];
                    const embeddingOutput = await embedder(chunk);

                    let embedding;
                    if (embeddingOutput && typeof embeddingOutput === 'object' && 'data' in embeddingOutput) {
                        embedding = Array.from(embeddingOutput.data);
                    } else if (Array.isArray(embeddingOutput)) {
                        embedding = embeddingOutput.flat(Infinity);
                    } else if (embeddingOutput && typeof embeddingOutput === 'object') {
                        embedding = Object.values(embeddingOutput);
                    } else {
                        console.warn(`⚠️ Unexpected output format for ${file} chunk ${i}`);
                        continue;
                    }

                    const resized = resizeEmbedding(embedding);
                    embeddingsMap[file].push({ chunkId: i, embedding, text: chunk });
                    console.log(`✅ Chunk ${i} of ${file} embedded (length: ${resized.length})`);
                }
            } catch (err) {
                console.error(`❌ Failed processing ${file}:`, err);
            }
        }

        if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
            console.log("🔌 Connecting to Chroma Cloud...");
            const client = new CloudClient({
                apiKey: CHROMA_API_KEY,
                tenant: CHROMA_TENANT,
                database: CHROMA_DATABASE
            });

            let collection;
            try {
                collection = await client.getCollection({ name: COLLECTION_NAME });
                console.log("✅ Коллекция найдена:", collection.name);
            } catch {
                collection = await client.createCollection({ name: COLLECTION_NAME });
                console.log("✅ Коллекция создана:", collection.name);
            }

            // Пушим эмбеддинги с текстом
            for (const [file, chunks] of Object.entries(embeddingsMap)) {
                for (const { chunkId, embedding, text } of chunks) {
                    try {
                        await collection.add({
                            ids: [makeId(file, chunkId)],
                            embeddings: [embedding],
                            metadatas: [{ file, chunkId, text }]
                        });
                        console.log(`✅ Pushed ${file} part ${chunkId} to Chroma Cloud`);
                    } catch (err) {
                        console.error(`❌ Failed to push ${file} part ${chunkId}:`, err.message);
                    }
                }
            }
        } else {
            console.log("⚠️ Chroma credentials not provided, пропускаем upload.");
        }

        console.log("✅ Vectorization complete");

        // Пример поиска
        const ISSUE_BODY = process.env.ISSUE_BODY || "function App()";
        console.log(`🔍 Vectorizing issue text: "${ISSUE_BODY}"`);
        const issueEmbeddingOutput = await embedder(ISSUE_BODY);
        const issueEmbedding = resizeEmbedding(Array.isArray(issueEmbeddingOutput) ? issueEmbeddingOutput.flat(Infinity) : Array.from(issueEmbeddingOutput.data));
        console.log("✅ Issue text embedded");

        console.log("🔎 Querying Chroma for relevant chunks...");
        const results = await client.getCollection({ name: COLLECTION_NAME }).then(col =>
            col.query({
                queryEmbeddings: [issueEmbedding],
                nResults: 5,
                include: ["metadatas", "distances"]
            })
        );

        console.log("✅ Top relevant chunks:");
        const metadatas = results.metadatas?.[0] || [];
        const distances = results.distances?.[0] || [];
        for (let i = 0; i < metadatas.length; i++) {
            const meta = metadatas[i];
            const snippet = meta?.text ? meta.text.slice(0, 200).replace(/\n/g, ' ') : '[no text]';
            console.log(`- File: ${meta?.file}, chunk: ${meta?.chunkId}, distance: ${distances[i]?.toFixed(2)}, snippet: ${snippet}`);
        }

    } catch (err) {
        console.error("❌ Fatal error:", err);
    }
}

main();










