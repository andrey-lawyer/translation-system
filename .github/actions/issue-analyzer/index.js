import { CloudClient } from "chromadb";
import { pipeline } from "@xenova/transformers";
import { Octokit } from "@octokit/rest";

// Environment validation
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
const RETRY_DELAY = 2000; // 2 seconds

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
            if (i < retries - 1) {
                await sleep(delay * (i + 1));
            }
        }
    }
    throw lastError;
}

async function aiGenerateFix(issueText, relevantChunks) {
    try {
        if (!relevantChunks || relevantChunks.length === 0) {
            throw new Error('No relevant chunks provided');
        }

        // TODO: Implement actual AI processing here
        const chunk = relevantChunks[0];
        if (!chunk.metadata || !chunk.metadata.file) {
            throw new Error('Invalid chunk metadata');
        }

        return {
            file: chunk.metadata.file,
            newContent: `// Auto-generated fix for issue #${ISSUE_NUMBER}\n${chunk.text}`
        };
    } catch (error) {
        console.error('Error in AI generation:', error);
        throw error;
    }
}

async function main() {
    try {
        console.log('🚀 Starting issue analysis...');

        // 1️⃣ Vectorize Issue text
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
            try {
                const col = await client.getCollection({ name: COLLECTION_NAME });
                console.log(`✅ Connected to collection: ${col.name}`);
                return col;
            } catch (error) {
                console.error('❌ Collection not found:', error.message);
                throw error;
            }
        });

        // 3️⃣ Query relevant chunks
        console.log('🔎 Searching for relevant code...');
        const results = await withRetry(async () => {
            const res = await collection.query({
                query_embeddings: [issueEmbedding],
                n_results: 3
            });

            if (!res || !res.documents || res.documents.length === 0) {
                throw new Error('No results from Chroma');
            }
            return res;
        });

        // 4️⃣ Generate patch
        console.log('🤖 Generating fix...');
        const patch = await aiGenerateFix(ISSUE_BODY, results[0].documents.map((text, idx) => ({
            text,
            metadata: results[0].metadatas[idx]
        })));

        // 5️⃣ Create branch and commit
        console.log('🌿 Creating branch...');
        const [owner, repo] = GITHUB_REPOSITORY.split('/');
        const octokit = new Octokit({
            auth: GITHUB_TOKEN,
            request: { timeout: 10000 } // 10s timeout
        });

        const branchName = `issue-${ISSUE_NUMBER}`;

        // Get main branch SHA
        const { data: mainRef } = await withRetry(() =>
            octokit.git.getRef({
                owner,
                repo,
                ref: "heads/main"
            })
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

        // Get file SHA if exists
        let fileSha;
        try {
            const { data: fileData } = await octokit.repos.getContent({
                owner,
                repo,
                path: patch.file
            });
            fileSha = fileData.sha;
        } catch (error) {
            if (error.status !== 404) { // 404 means file doesn't exist yet
                throw error;
            }
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
