## Dataset Construction and AI Text Generation

To create the AI-generated counterparts, we used four different large language models: GPT-4o-mini, Gemini 1.5 Flash, Claude-3-Haiku, and DeepSeek-v3. These models were accessed through different API platforms during data generation.

The original cleaned human dataset was divided into four equal and mutually exclusive subsets. Each subset was assigned to exactly one AI model, and the four generation scripts were run simultaneously. This design ensured balanced coverage across generators while avoiding unnecessary duplication.

After generation was completed, the four AI-generated subsets were manually merged with the original human dataset to construct the final unified dataset.

As a result, the final dataset contains:
- Human-written statements
- AI-generated counterparts
- Four-way labels: `Human-Real`, `Human-Fake`, `AI-Real`, `AI-Fake`
- Original metadata fields
