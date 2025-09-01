import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";

// Chroma credentials
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';

const COLLECTION_NAME = "FullProjectCollection";
const MAX_CHUNK_LENGTH = 1000; // символов на блок
const MAX_EMBED_DIM = 3072;    // лимит для Chroma Starter
const ALLOWED_EXTENSIONS = ['.js', '.ts', '.go', '.groovy', '.html', '.css', '.md'];
const EXCLUDED_FOLDERS = ['node_modules', 'target', 'dist', '.git'];

// ======= UTILS =======

// Создание короткого уникального ID для Chroma
function makeId(file, chunkId) {
    const hash = crypto.createHash('sha256').update(file).digest('hex').slice(0, 12);
    return `${hash}_p${chunkId}`;
}

// Разбиваем текст на чанки
function splitText(text, maxLength) {
    const chunks = [];
    for (let i = 0; i < text.length; i += maxLength) {
        chunks.push(text.slice(i, i + maxLength));
    }
    return chunks;
}

// Уменьшаем размерность эмбеддинга до лимита
function resizeEmbedding(embedding, maxDim = MAX_EMBED_DIM) {
    if (embedding.length <= maxDim) return embedding;
    const factor = embedding.length / maxDim;
    const resized = [];
    for (let i = 0; i < maxDim; i++) {
        const idx = Math.floor(i * factor);
        resized.push(embedding[idx]);
    }
    return resized;
}

// Рекурсивно собираем все файлы с нужными расширениями, исключая папки
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

// ======= MAIN =======
async function main() {
    console.log("Загружаем локальную модель эмбеддингов...");
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        quantized: true,
        revision: 'main',
        device: 'cpu',
        pooling: 'mean',
        normalize: true
    });
    console.log("Модель загружена ✅");

    const projectRoot = path.resolve('./'); // корень проекта
    const filesToVectorize = await getFiles(projectRoot);
    console.log(`Найдено файлов для векторизации: ${filesToVectorize.length}`);

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
                    console.warn(`Unexpected output format for ${file} chunk ${i}`);
                    continue;
                }

                const resized = resizeEmbedding(embedding);
                embeddingsMap[file].push({ chunkId: i, embedding: resized });
                console.log(`Chunk ${i} of ${file} embedded (length: ${resized.length})`);
            }
        } catch (err) {
            console.error(`❌ Failed processing ${file}:`, err);
        }
    }

    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        try {
            console.log("Connecting to Chroma Cloud...");
            const client = new CloudClient({
                apiKey: CHROMA_API_KEY,
                tenant: CHROMA_TENANT,
                database: CHROMA_DATABASE,
            });

            // Получаем или создаём коллекцию
            let collection;
            try {
                collection = await client.getCollection({ name: COLLECTION_NAME });
                console.log("Коллекция найдена:", collection.name, collection.id);
            } catch {
                collection = await client.createCollection({ name: COLLECTION_NAME });
                console.log("Коллекция создана:", collection.name, collection.id);
            }

            // Пушим эмбеддинги
            for (const [file, chunks] of Object.entries(embeddingsMap)) {
                for (const { chunkId, embedding } of chunks) {
                    try {
                        await collection.add({
                            ids: [makeId(file, chunkId)],
                            embeddings: [embedding],
                            metadatas: [{ file, chunkId }],
                        });
                        console.log(`✅ Pushed ${file} part ${chunkId} to Chroma Cloud`);
                    } catch (err) {
                        console.error(`❌ Failed to push ${file} part ${chunkId}:`, err.message);
                    }
                }
            }
        } catch (err) {
            console.error("Chroma Cloud connection failed:", err.message);
        }
    } else {
        console.log("Chroma Cloud credentials not provided, пропускаем upload.");
    }

    console.log("Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));








