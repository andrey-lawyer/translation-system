import { CloudClient } from "chromadb";
import { pipeline } from "@xenova/transformers";
import { Octokit } from "@octokit/rest";
import fs from "fs/promises";
import path from "path";

// ====== ENVIRONMENT VALIDATION ======
const requiredEnvVars = [
    'CHROMA_API_KEY',
    'CHROMA_TENANT',
    'CHROMA_DATABASE',
    'ISSUE_BODY',
    'ISSUE_NUMBER',
    'GITHUB_TOKEN',
    'GITHUB_REPOSITORY'
];

for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
        console.error(`❌ Missing required environment variable: ${envVar}`);
        process.exit(1);
    }
}

const {
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    ISSUE_BODY,
    ISSUE_NUMBER,
    GITHUB_TOKEN,
    GITHUB_REPOSITORY
} = process.env;

const COLLECTION_NAME = "FullProjectCollection";
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // ms

// ====== UTILS ======
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(fn, retries = MAX_RETRIES, delay = RETRY_DELAY) {
    let lastError;
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (err) {
            lastError = err;
            console.warn(`Attempt ${i + 1} failed: ${err.message}`);
            if (i < retries - 1) await sleep(delay * (i + 1));
        }
    }
    throw lastError;
}

// ====== MAIN ======
async function main() {
    try {
        console.log("🚀 Starting issue analysis...");

        // 1️⃣ Vectorize issue
        console.log("🔍 Vectorizing issue text...");
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true,
            device: "cpu",
            pooling: "mean",
            normalize: true
        });

        let issueEmbedding;
        const embeddingOutput = await embedder(ISSUE_BODY);

        if (embeddingOutput && 'data' in embeddingOutput) {
            // pooling: "mean" даёт сразу вектор 3072
            issueEmbedding = Array.from(embeddingOutput.data);
        } else if (Array.isArray(embeddingOutput) && embeddingOutput.length && Array.isArray(embeddingOutput[0])) {
            // fallback: усредняем вручную
            const sum = new Array(embeddingOutput[0].length).fill(0);
            for (const tokenVec of embeddingOutput) {
                for (let i = 0; i < tokenVec.length; i++) sum[i] += tokenVec[i];
            }
            issueEmbedding = sum.map(x => x / embeddingOutput.length);
        } else {
            throw new Error("Unexpected embedding output format");
        }

        console.log("✅ Issue embedding ready (length:", issueEmbedding.length, ")");

        // 2️⃣ Connect to Chroma
        console.log("🔌 Connecting to Chroma...");
        const client = new CloudClient({
            apiKey: CHROMA_API_KEY,
            tenant: CHROMA_TENANT,
            database: CHROMA_DATABASE
        });

        const collection = await withRetry(async () => {
            const col = await client.getCollection({ name: COLLECTION_NAME });
            console.log(`✅ Connected to collection: ${col.name}`);
            return col;
        });

        // 3️⃣ Query relevant chunks
        console.log("🔎 Searching for relevant code...");
        const results = await withRetry(async () => {
            const res = await collection.query({
                queryEmbeddings: [issueEmbedding],
                nResults: 5,
                include: ["metadatas", "distances"]
            });

            if (!res || !res.metadatas || res.metadatas.length === 0) {
                throw new Error("No results from Chroma");
            }
            return res;
        });

        const metadatas = results.metadatas[0];
        const distances = results.distances?.[0];
        const relevantChunks = [];

        console.log("✅ Found relevant chunks:");
        for (let i = 0; i < metadatas.length; i++) {
            const meta = metadatas[i];
            const file = meta?.file;
            const chunkId = meta?.chunkId ?? 0;
            let chunkText = "[text not available]";

            if (file) {
                try {
                    const content = await fs.readFile(path.resolve(file), "utf-8");
                    const chunkSize = 1000; // такой же, как при загрузке
                    const start = chunkId * chunkSize;
                    const end = start + chunkSize;
                    chunkText = content.slice(start, end).replace(/\n/g, "\\n");
                } catch (err) {
                    console.warn(`Failed to read chunk ${chunkId} from ${file}: ${err.message}`);
                }
            }

            const distance = distances?.[i]?.toFixed(2) ?? "n/a";
            console.log(`- [${file} | chunk ${chunkId} | distance: ${distance}] -> ${chunkText}`);
            relevantChunks.push({ text: chunkText, metadata: meta });
        }

        // 4️⃣ Generate patch (simple AI simulation)
        console.log("🤖 Generating fix...");
        const patch = {
            file: relevantChunks[0]?.metadata?.file,
            newContent: `// Auto-generated fix for issue #${ISSUE_NUMBER}\n${relevantChunks[0]?.text || ''}`
        };

        if (!patch.file) throw new Error("No valid file found for patch");

        // 5️⃣ GitHub: create branch, commit, PR
        console.log("🌿 Creating branch and PR...");
        const [owner, repo] = GITHUB_REPOSITORY.split('/');
        const octokit = new Octokit({ auth: GITHUB_TOKEN });

        const branchName = `issue-${ISSUE_NUMBER}`;
        const { data: mainRef } = await withRetry(() =>
            octokit.git.getRef({ owner, repo, ref: "heads/main" })
        );

        // Create branch
        await withRetry(() =>
            octokit.git.createRef({
                owner,
                repo,
                ref: `refs/heads/${branchName}`,
                sha: mainRef.object.sha
            })
        );
        console.log(`✅ Branch ${branchName} created`);

        // Check if file exists
        let fileSha;
        try {
            const { data: fileData } = await octokit.repos.getContent({ owner, repo, path: patch.file });
            fileSha = fileData.sha;
        } catch (err) {
            if (err.status !== 404) throw err;
        }

        // Commit file
        await withRetry(() =>
            octokit.repos.createOrUpdateFileContents({
                owner,
                repo,
                path: patch.file,
                message: `Fix for issue #${ISSUE_NUMBER}`,
                content: Buffer.from(patch.newContent).toString('base64'),
                sha: fileSha,
                branch: branchName
            })
        );
        console.log(`✅ Changes committed to ${patch.file}`);

        // Create PR
        const pr = await withRetry(() =>
            octokit.pulls.create({
                owner,
                repo,
                title: `Fix for issue #${ISSUE_NUMBER}`,
                head: branchName,
                base: "main",
                body: `This is an automated fix for issue #${ISSUE_NUMBER}`
            })
        );
        console.log(`✅ Pull request created: ${pr.data.html_url}`);

        console.log("✅ All done!");

    } catch (err) {
        console.error("❌ Fatal error:", err);
        process.exit(1);
    }
}

main().catch(err => {
    console.error("Unhandled error:", err);
    process.exit(1);
});

