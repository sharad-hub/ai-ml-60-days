# The 60-Day AI/ML Fast-Track: Project-First Roadmap

**The shift from the 100-day version:** almost no upfront theory. You get ~3 days of "just enough to be dangerous," then you're inside Project 1, and every new concept gets introduced *the moment the project needs it* — not before. You'll Google/read about a concept for 20 minutes, then immediately use it. Concepts get revisited across projects, which is how they actually stick.

**5 end-to-end projects, each shipped (not just "completed in a notebook"):**

| # | Project | Days | What "shipped" means |
|---|---|---|---|
| 1 | Movie Rating Predictor | 4-13 | A CLI/Streamlit tool that predicts ratings from features, with a real metrics report |
| 2 | Poster Genre Classifier | 14-25 | A CNN (transfer learning) that classifies movie posters, wrapped in a simple web app |
| 3 | Review Sentiment + Summarizer | 26-36 | A fine-tuned Transformer model deployed as an API |
| 4 | "Ask Signal" RAG Chatbot | 37-49 | A working chatbot that answers questions over your movie dataset, grounded + cited |
| 5 | Signal — the unified app | 50-60 | All 4 above combined, containerized, deployed live with a public URL |

Same domain (movies) throughout so you're not re-learning a new dataset every time — the *dataset* stays familiar so all your new energy goes into the *technique*.

---

## Days 1-3: Just Enough to Be Dangerous

Don't try to master this — just get functional. You'll fill gaps in Project 1.

| Day | Do This |
|---|---|
| 1 | Set up Python, VS Code/Jupyter, GitHub repo `ai-ml-60-days`. Speed-run Python essentials (functions, loops, dicts, classes) — 2 hrs max. Load a CSV with Pandas, print `.head()`, `.info()`, `.describe()`. |
| 2 | Pandas + NumPy crash session: filtering, groupby, merge, basic array ops. Do this *on the actual movie dataset* (MovieLens or TMDB) you'll use for the whole roadmap. |
| 3 | 90-minute crash intro to what ML "is": features → model → prediction, train/test split, what overfitting means (concept only, one diagram, move on). Install scikit-learn + PyTorch. |

**Motivation checkpoint:** by end of Day 3 you should have a GitHub repo with 3 commits and a dataset you can already talk about. That's real progress — don't let "I don't know enough yet" creep in.

---

## Project 1 (Days 4-13): Movie Rating Predictor — Classical ML End-to-End

You will touch regression, classification, feature engineering, evaluation, and model comparison — all inside one shipped project.

| Day | Concept introduced (just-in-time) | Hands-On |
|---|---|---|
| 4 | Regression basics + `train_test_split` | Build a first (bad) `LinearRegression` model predicting rating from 2-3 numeric features. It working badly is fine — that's the hook for tomorrow. |
| 5 | Feature engineering | Add engineered features (decade, genre count, log-budget). Re-measure — show yourself the improvement. |
| 6 | Evaluation metrics (MAE/RMSE/R²) | Build a reusable `evaluate(model, X, y)` function. Use it from now on for everything. |
| 7 | Classification + logistic regression | Reframe as "hit vs flop" binary classification. Add precision/recall/F1/confusion matrix. |
| 8 | Decision trees + random forests | Train both, compare against your linear/logistic baselines in one table. |
| 9 | Gradient boosting (XGBoost) | Train XGBoost. It should win. Understand *why* boosting tends to win on tabular data (10-min read, not more). |
| 10 | Cross-validation + hyperparameter tuning | Tune your best model with `RandomizedSearchCV` or Optuna. Log the before/after score. |
| 11 | sklearn Pipelines | Refactor everything into one clean `Pipeline` + `ColumnTransformer`. Messy notebook → one script. |
| 12 | Wrapping a model for use (serialization) | Save the pipeline with `joblib`. Write a `predict.py` that loads it and takes CLI input. |
| 13 | **Ship Day** | Wrap `predict.py` in a 1-page Streamlit app: user enters movie features, gets a predicted rating + hit probability. Push, write a short README with your model comparison table. **This is shippable project #1 — post it somewhere (LinkedIn/Twitter/a friend).** |

---

## Project 2 (Days 14-25): Poster Genre Classifier — Deep Learning + Computer Vision

| Day | Concept introduced (just-in-time) | Hands-On |
|---|---|---|
| 14 | What a neural net is (perceptron → MLP, 1 diagram, no derivations) | Build a tiny MLP in PyTorch on a toy dataset just to see training loop mechanics. |
| 15 | Backprop & gradient descent (intuition, not derivation) | Train the same MLP longer; plot the loss curve going down. Explain in your log, in your own words, what's happening. |
| 16 | Images as tensors + what convolution does | Run one hand-written convolution filter (edge detector) on an image manually with NumPy. |
| 17 | CNN architecture (conv + pool layers) | Build a small CNN in PyTorch; train on CIFAR-10 subset to confirm it works before touching your real data. |
| 18 | Collecting your real dataset | Pull ~1000 movie posters + genre labels via TMDB API. Do basic cleaning. |
| 19 | Transfer learning (the single highest-leverage CV technique) | Load pretrained ResNet18, freeze base, replace final layer for your genre classes. |
| 20 | Data augmentation | Add flip/rotate/color-jitter; compare validation accuracy with/without. |
| 21 | Training + fine-tuning | Train fully, unfreeze last few layers, fine-tune with a small learning rate. |
| 22 | Multi-class evaluation | Build a confusion matrix; identify which genres get confused (this is usually the most interesting part). |
| 23 | Model interpretability (Grad-CAM) | Visualize what the model "looks at" on 5 posters. This is a great share-on-LinkedIn visual. |
| 24 | Exporting the model | Export with TorchScript or just `.pt` + inference script; test loading fresh. |
| 25 | **Ship Day** | Small web app (Streamlit/Gradio): upload a poster image → get predicted genre + confidence + Grad-CAM overlay. Push + write up. **Shippable project #2.** |

---

## Project 3 (Days 26-36): Sentiment + Summarizer — NLP & Transformers

| Day | Concept introduced (just-in-time) | Hands-On |
|---|---|---|
| 26 | Text preprocessing + why raw text ≠ numbers | Clean a set of movie reviews (IMDB dataset). |
| 27 | TF-IDF baseline | Build a fast logistic regression sentiment classifier on TF-IDF. This is your baseline to beat. |
| 28 | Word embeddings (concept: words as vectors) | Load pretrained GloVe; find nearest neighbors of movie-related words. Quick, visual, satisfying. |
| 29 | What attention does (intuition only) | Read "The Illustrated Transformer" (1 hr, this is the highest-value read in the whole roadmap). |
| 30 | Hugging Face `transformers` basics | Load pretrained DistilBERT; run inference on 5 reviews, no training yet. |
| 31 | Fine-tuning a Transformer | Fine-tune DistilBERT on your review sentiment data. |
| 32 | Compare all 3 approaches | TF-IDF vs. embeddings-based vs. fine-tuned BERT — one comparison table, explain the jumps. |
| 33 | Summarization with pretrained models | Use BART/T5 to summarize long reviews — zero training needed, just correct usage. |
| 34 | Tokenizers (just enough to not be confused later) | Inspect how BPE/WordPiece splits your own sentences. 30 min max. |
| 35 | Semantic search (embeddings + cosine similarity) | Embed movie descriptions; build "find similar movies." This directly sets up Project 4. |
| 36 | **Ship Day** | API (FastAPI) with `/sentiment`, `/summarize`, `/similar-movies` endpoints. Test with real requests. **Shippable project #3.** |

---

## Project 4 (Days 37-49): "Ask Signal" — RAG Chatbot with LLMs

| Day | Concept introduced (just-in-time) | Hands-On |
|---|---|---|
| 37 | LLM landscape + calling an API programmatically | Script that sends a movie question to an LLM API, parses the response. |
| 38 | Vector databases (Chroma/FAISS) | Store your Day 35 embeddings in a real vector DB instead of a plain list. |
| 39 | RAG architecture: retrieval half | Given a question, retrieve top-5 relevant movie docs. |
| 40 | RAG: generation half | Feed retrieved docs + question to the LLM. Compare grounded vs. ungrounded answers side by side — this contrast is the "aha" moment of RAG. |
| 41 | Chunking strategy | Try 2-3 chunk sizes; see how retrieval quality changes. |
| 42 | Evaluating RAG quality | Build a 15-question eval set; score accuracy/faithfulness by hand. |
| 43 | Tool/function calling | Give the LLM a "query_database" tool for things pure retrieval can't answer (e.g. "average rating of 2015 movies"). |
| 44 | Basic agent loop (ReAct-style) | LLM decides: search or answer directly; observes result; responds. |
| 45 | Guardrails | Add "cite your source" + a graceful "I don't know" fallback. |
| 46 | Conversation memory | Support follow-up questions using chat history. |
| 47 | Chat UI | Wrap it in Streamlit/Gradio chat interface. |
| 48 | Cost/latency pass | Add caching for repeat queries; measure the improvement. |
| 49 | **Ship Day** | Full "Ask Signal" chatbot live, answering grounded questions about your movie dataset with citations. **Shippable project #4 — this is usually the most impressive demo of the whole roadmap.** |

---

## Project 5 (Days 50-60): Signal — Unify, Deploy, Ship for Real

| Day | Concept introduced (just-in-time) | Hands-On |
|---|---|---|
| 50 | API design for multi-model apps | Put all 4 projects' models behind one FastAPI app: `/predict-rating`, `/classify-poster`, `/sentiment`, `/ask`. |
| 51 | Testing ML APIs | Write basic pytest tests for each endpoint, including bad-input cases. |
| 52 | Docker basics | Containerize the whole app; confirm it runs identically from the container. |
| 53 | Secrets/env management | Move API keys to `.env`; document setup clearly in the README. |
| 54 | One unified frontend | Build a single clean page/app that hits all 4 endpoints — this is your portfolio centerpiece. |
| 55 | Deployment | Deploy on a free tier (Render/Railway/HF Spaces/Fly.io). Get a public URL. |
| 56 | Logging & monitoring basics | Add request logging + a `/metrics` endpoint (count, latency, errors). |
| 57 | Bug bash + polish | Use the live app for 30 min like a stranger would. Fix the 5 most annoying rough edges. |
| 58 | Write-up | Architecture diagram + README explaining what Signal does and how it's built. |
| 59 | Public share | Post it: LinkedIn, Twitter/X, a relevant subreddit, or just to 5 people directly. Ask for feedback. |
| 60 | **Retrospective** | Review your Day 1 log vs. now. Write: what surprised you, your favorite of the 5 projects, and pick your next 30-day focus (Kaggle, a specific subfield, contributing to open source, going deeper on one of these 4 techniques). |

---

## Staying Motivated for 60 Straight Days

**1. Make "shipped" visible, every ~10-13 days.**
This roadmap is built around 5 concrete ship dates, not 60 days of vague progress. A shipped thing — even small — gives you a dopamine hit a notebook never will. Actually post or show each one somewhere, even to one person. External visibility is a stronger motivator than internal willpower.

**2. Track a streak, but forgive it.**
Commit to GitHub daily. But if you miss a day, the rule is: *never miss twice*. One skipped day is life; two in a row is how roadmaps die. Just restart the next day at the exact spot you left off — no "re-doing" earlier days out of guilt.

**3. Keep a 3-line daily log, not a diary.**
Each day, write: (1) what you built, (2) one thing that broke and how you fixed it, (3) one thing you now understand that you didn't yesterday. This log becomes your proof-of-progress on low-motivation days — reread it when you doubt you're improving.

**4. Timebox the "boring" parts.**
Reading/theory gets a hard cap (noted in the plan — usually under an hour). If a concept isn't clicking in that time, move on and build anyway; understanding often arrives *while* debugging, not while reading.

**5. Lower the daily bar on hard days.**
The real rule isn't "3 hours a day," it's "don't break the chain." On a low-energy day, the win condition is: touch the code, make one commit, even a 15-minute one. Momentum matters more than intensity on any single day.

**6. Anchor motivation to the project, not the discipline.**
"I'm learning ML" is abstract and easy to abandon. "I'm building Signal and today I make its chatbot cite sources" is concrete and interesting. Every day, reread that day's Hands-On task before you start — it's a specific, finishable thing, not homework.

**7. Get one accountability point of contact.**
A friend, a Discord/Reddit ML community, or just a public repo/Twitter thread where you post daily. Knowing one other person might glance at Day 37's commit is enough pressure to actually do Day 37.

**8. Review weekly, not daily.**
Every 7th day, skim the week's log instead of learning something new. Redo the one exercise that felt shakiest. This is also a natural rest point that prevents burnout without breaking the streak.

**9. Remember why fast-paced works for you.**
You chose speed + hands-on because momentum and shipping create interest — protect that by resisting the urge to "go back and fully master" something before moving on. Depth comes from *reusing* a concept across all 5 projects (you'll use evaluation metrics, train/test splits, and embeddings in almost every project) — not from perfecting it once in isolation.

**10. The finish line is a live URL with your name on it.**
By Day 60 you'll have a deployed, working AI product, not just "knowledge." Keep that end image in mind on the days motivation dips — you're not just learning, you're building something that will exist and that you can point to.
