import { CloudClient } from "chromadb";
import { pipeline } from "@xenova/transformers";
import { Octokit } from "@octokit/rest";

// ====== ENV VALIDATION ======
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
const RETRY_DELAY = 2000;

// ====== UTILS ======
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(fn, retries = MAX_RETRIES, delay = RETRY_DELAY) {
    let lastError;
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            console.warn(`Attempt ${i + 1} failed:`, error.message);
            if (i < retries - 1) await sleep(delay * (i + 1));
        }
    }
    throw lastError;
}

// ====== AI FIX GENERATOR ======
async function aiGenerateFix(issueText, relevantChunks) {
    if (!relevantChunks || relevantChunks.length === 0) {
        throw new Error('No relevant chunks provided');
    }

    const chunk = relevantChunks[0];
    if (!chunk.metadata || !chunk.metadata.file) {
        throw new Error('Invalid chunk metadata');
    }

    return {
        file: chunk.metadata.file,
        newContent: `// Auto-generated fix for issue #${ISSUE_NUMBER}\n${chunk.text || "// [text not available]"}`
    };
}

// ====== MAIN ======
async function main() {
    try {
        console.log('🚀 Starting issue analysis...');

        // 1️⃣ Vectorize issue text
        console.log('🔍 Vectorizing issue text...');
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true,
            device: "cpu",
            pooling: "mean",
            normalize: true
        });

        const issueEmbedding = await withRetry(async () => {
            const output = await embedder(ISSUE_BODY);
            if (!output || !output.data) {
                throw new Error('Invalid embedding output');
            }
            return Array.from(output.data);
        });

        // 2️⃣ Connect to Chroma
        console.log('🔌 Connecting to Chroma...');
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
        console.log('🔎 Searching for relevant code...');
        const results = await withRetry(async () => {
            const res = await collection.query({
                queryEmbeddings: [issueEmbedding], // правильно с большой E
                nResults: 5,                        // правильно с большой R
                include: ["metadatas", "distances"] // documents не нужны
            });

            if (!res || !res.metadatas || res.metadatas.length === 0) {
                throw new Error('No results from Chroma');
            }
            return res;
        });

        const metadatas = results.metadatas[0];
        const distances = results.distances?.[0];

        // Подготовка списка для AI генерации
        const relevantChunks = metadatas.map((meta, idx) => ({
            metadata: meta,
            text: `[text not available]`, // документы не сохранялись
            distance: distances?.[idx]?.toFixed(2) ?? "n/a"
        }));

        console.log("✅ Found relevant chunks:");
        relevantChunks.forEach(chunk => {
            console.log(`- [${chunk.metadata.file} | chunk ${chunk.metadata.chunkId} | distance: ${chunk.distance}] -> ${chunk.text}`);
        });

        // 4️⃣ Generate patch
        console.log('🤖 Generating fix...');
        const patch = await aiGenerateFix(ISSUE_BODY, relevantChunks);

        // 5️⃣ Create branch and commit
        console.log('🌿 Creating branch...');
        const [owner, repo] = GITHUB_REPOSITORY.split('/');
        const octokit = new Octokit({ auth: GITHUB_TOKEN, request: { timeout: 10000 } });
        const branchName = `issue-${ISSUE_NUMBER}`;

        const { data: mainRef } = await withRetry(() =>
            octokit.git.getRef({ owner, repo, ref: "heads/main" })
        );

        await withRetry(() =>
            octokit.git.createRef({ owner, repo, ref: `refs/heads/${branchName}`, sha: mainRef.object.sha })
        );

        // Get file SHA if exists
        let fileSha;
        try {
            const { data: fileData } = await octokit.repos.getContent({ owner, repo, path: patch.file });
            fileSha = fileData.sha;
        } catch (error) {
            if (error.status !== 404) throw error;
        }

        // Commit changes
        console.log('💾 Committing changes...');
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

        // Create PR
        console.log('📝 Creating pull request...');
        await withRetry(() =>
            octokit.pulls.create({
                owner,
                repo,
                title: `Fix for issue #${ISSUE_NUMBER}`,
                head: branchName,
                base: 'main',
                body: `This is an automated fix for issue #${ISSUE_NUMBER}`
            })
        );

        console.log('✅ All done!');
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

main().catch(err => {
    console.error('Unhandled error:', err);
    process.exit(1);
});
