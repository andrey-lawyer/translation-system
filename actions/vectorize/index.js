import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { CloudClient } from "chromadb";

// Инициализация OpenAI и Chroma Cloud
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE
});

// Функция для обработки файла
async function embedFile(filePath) {
    const content = fs.readFileSync(filePath, "utf-8");
    if (!content.trim()) return;

    const lines = content.split("\n");
    for (let i = 0; i < lines.length; i += 200) {
        const chunk = lines.slice(i, i + 200).join("\n");

        const embedding = (await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: chunk
        })).data[0].embedding;

        const collection = await chroma.getCollection({ name: "translation-system" }).catch(async () => {
            return await chroma.createCollection({ name: "translation-system" });
        });

        await collection.add({
            ids: [`${filePath}-${i}`],
            embeddings: [embedding],
            documents: [chunk],
            metadatas: [{ file: filePath, start_line: i }]
        });

        console.log(`Indexed chunk ${i} of ${filePath}`);
    }
}

// Рекурсивный обход репозитория
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
