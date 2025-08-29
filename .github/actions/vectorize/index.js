import fs from "fs";
import path from "path";
import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";

const CHUNK_SIZE = 100; // меньше строк → меньше вероятность ошибок
const DELAY_MS = 200;

// Инициализация Chroma Cloud
const chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE
});

let embedder;

// Загрузка локальной модели эмбеддингов
async function initEmbedder() {
    console.log("Loading local embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("Model loaded ✅");
}

// Получение эмбеддинга для текста
async function getEmbedding(text) {
    if (!embedder) throw new Error("Embedder not initialized");

    const result = await embedder(text);

    // Усреднение токенов
    if (Array.isArray(result[0])) {
        const tokens = result[0];
        const vector = tokens[0].map((_, i) =>
            tokens.reduce((sum, t) => sum + t[i], 0) / tokens.length
        );
        return vector;
    }
    return result;
}

async function sleep(ms) {
    return new Promise((res) => setTimeout(res, ms));
}

// Индексация одного файла
async function embedFile(filePath) {
    const stats = fs.statSync(filePath);
    if (stats.size > 200_000) return; // слишком большой файл

    const content = fs.readFileSync(filePath, "utf-8");
    if (!content.trim()) return;

    const lines = content.split("\n");

    // Получаем или создаём коллекцию
    const collectionName = "translation-system";
    let collection;
    try {
        collection = await chroma.getCollection({ name: collectionName });
    } catch {
        collection = await chroma.createCollection({ name: collectionName });
    }

    for (let i = 0; i < lines.length; i += CHUNK_SIZE) {
        const chunk = lines.slice(i, i + CHUNK_SIZE).join("\n");
        if (!chunk.trim()) continue;

        try {
            const embedding = await getEmbedding(chunk);

            // Добавляем в Chroma Cloud
            await collection.add({
                ids: [`${filePath}-${i}`],
                embeddings: [embedding],
                documents: [chunk],
                metadatas: [{ file: filePath, start_line: i }],
            });

            console.log(`Indexed chunk ${i} of ${filePath}`);
            await sleep(DELAY_MS);
        } catch (err) {
            console.error(`Error embedding ${filePath} chunk ${i}:`, err.message || err);
        }
    }
}

// Обход директории рекурсивно
async function walk(dir) {
    const files = fs.readdirSync(dir, { withFileTypes: true });
    for (const file of files) {
        const filePath = path.join(dir, file.name);

        if (file.isDirectory()) {
            if ([".git", "node_modules", "dist", "target"].includes(file.name)) continue;
            await walk(filePath);
        } else if (/\.(go|java|ts|tsx|js|md)$/i.test(file.name)) {
            await embedFile(filePath);
        }
    }
}

// Основной запуск
(async () => {
    await initEmbedder();
    console.log("Starting vectorization (local embeddings + Chroma Cloud)...");
    await walk(".");
    console.log("✅ Vectorization complete");
})();



