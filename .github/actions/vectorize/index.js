import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";
import path from "path";

// Chroma credentials
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';

const COLLECTION_NAME = "FullProjectCollection";
const MAX_CHUNK_LENGTH = 1000; // символов на блок
const MAX_EMBED_DIM = 3072; // лимит для Chroma Starter

// Фильтр файлов
const INCLUDE_EXTENSIONS = [".js", ".ts", ".go", ".java", ".groovy", ".md", ".html", ".css"];
const EXCLUDE_DIRS = ["node_modules", "vendor", ".git", "build", "out"];

// Рекурсивно собираем все файлы с разрешёнными расширениями
async function getFiles(dir) {
    let files = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
            if (!EXCLUDE_DIRS.includes(entry.name)) {
                files = files.concat(await getFiles(fullPath));
            }
        } else if (INCLUDE_EXTENSIONS.includes(path.extname(entry.name))) {
            files.push(fullPath);
        }
    }

    return files;
}

// Разбиваем текст на чанки
function splitText(text, maxLength) {
    const chunks = [];
    for (let i = 0; i < text.length; i += maxLength) {
        chunks.push(text.slice(i, i + maxLength));
    }
    return chunks;
}

// Уменьшаем размерность эмбеддинга
function resizeEmbedding(embedding, maxDim = MAX_EMBED_DIM) {
    if (embedding.length <= maxDim) return embedding;
    const factor = embedding.length / maxDim;
    const resized = [];
    for (let i = 0; i < maxDim; i++) {
        resized.push(embedding[Math.floor(i * factor)]);
    }
    return resized;
}

async function getOrCreateCollection(client, name) {
    try {
        console.log("Ищем коллекцию по имени:", name);
        const collection = await client.getCollection({ name });
        console.log("Коллекция найдена:", collection.name, collection.id);
        return collection;
    } catch (err) {
        if (err.message.includes("collection not found") || err.name === "ChromaConnectionError") {
            console.log("Создаём новую коллекцию:", name);
            const collection = await client.createCollection({ name });
            console.log("Коллекция создана:", collection.name, collection.id);
            return collection;
        }
        console.error("Ошибка получения коллекции:", err.message);
        return null;
    }
}

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

    const projectFiles = await getFiles(process.cwd());
    console.log(`Найдено файлов для векторизации: ${projectFiles.length}`);

    const embeddingsMap = {};

    for (const file of projectFiles) {
        try {
            console.log(`Processing ${file}...`);
            const text = await fs.readFile(file, "utf-8");
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

            const collection = await getOrCreateCollection(client, COLLECTION_NAME);
            if (!collection) {
                console.warn("Коллекция недоступна. Пропускаем загрузку.");
                return;
            }

            for (const [file, chunks] of Object.entries(embeddingsMap)) {
                for (const { chunkId, embedding } of chunks) {
                    try {
                        await collection.add({
                            ids: [`${file}_part${chunkId}`],
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







