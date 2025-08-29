import fs from "fs";
import path from "path";
import { pipeline } from "@xenova/transformers";

// Chroma Cloud для хранения embeddings
import { CloudClient } from "chromadb";
const chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE
});

const CHUNK_SIZE = 500;
const DELAY_MS = 200;

// Инициализация локального embedder
let embedder;
(async () => {
    console.log("Loading local embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
})();

async function getEmbedding(text) {
    // Возвращает embedding как массив чисел
    const result = await embedder(text);
    // Если result — многомерный массив, усредняем токены
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
    return new Promise(res => setTimeout(res, ms));
}

async function embedFile(filePath) {
    const stats = fs.statSync(filePath);
    if (stats.size > 200_000) return;

    const content = fs.readFileSync(filePath, "utf-8");
    if (!content.trim()) return;

    const lines = content.split("\n");

    for (let i = 0; i < lines.length; i += CHUNK_SIZE) {
        const chunk = lines.slice(i, i + CHUNK_SIZE).join("\n");

        try {
            const embedding = await getEmbedding(chunk);

            const collection = await chroma.getCollection({ name: "translation-system" })
                .catch(async () => await chroma.createCollection({ name: "translation-system" }));

            await collection.add({
                ids: [`${filePath}-${i}`],
                embeddings: [embedding],
                documents: [chunk],
                metadatas: [{ file: filePath, start_line: i }]
            });

            console.log(`Indexed chunk ${i} of ${filePath}`);
            await sleep(DELAY_MS);
        } catch (err) {
            console.error(`Error embedding ${filePath} chunk ${i}:`, err.message || err);
        }
    }
}

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

(async () => {
    console.log("Starting vectorization (local embeddings)...");
    await walk(".");
    console.log("✅ Vectorization complete");
})();

