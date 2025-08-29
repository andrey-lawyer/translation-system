import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { CloudClient } from "chromadb";

// Настройки
const CHUNK_SIZE = 500; // строк на embedding
const DELAY_MS = 200;   // задержка между запросами

// Инициализация OpenAI и Chroma Cloud
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE
});

async function sleep(ms) {
    return new Promise((res) => setTimeout(res, ms));
}

async function embedFile(filePath) {
    const stats = fs.statSync(filePath);
    if (stats.size > 200_000) {
        console.log(`Skipping large file: ${filePath}`);
        return;
    }

    const content = fs.readFileSync(filePath, "utf-8");
    if (!content.trim()) return;

    const lines = content.split("\n");

    for (let i = 0; i < lines.length; i += CHUNK_SIZE) {
        const chunk = lines.slice(i, i + CHUNK_SIZE).join("\n");

        try {
            const embedding = (await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: chunk
            })).data[0].embedding;

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

// Запуск
(async () => {
    console.log("Starting vectorization...");
    await walk(".");
    console.log("✅ Vectorization complete");
})();
