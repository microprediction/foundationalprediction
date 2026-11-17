# SSRN submission: On the Importance of Evaluating Foundational Time-Series Models One Step Ahead in an Investment Context

Upload `horizon-note.pdf` (compiled from `horizon-note.tex` with `tectonic horizon-note.tex`; the served copy is https://foundationalprediction.com/horizon-note.pdf).

**Title.** On the Importance of Evaluating Foundational Time-Series Models One Step Ahead in an Investment Context

**Author.** Peter Cotton (ORCID 0000-0003-1832-2924), Microprediction

**Date.** November 2026

**Abstract** (plain text, 199 words).

Popular benchmarks for foundational time-series models pool forecast errors over long horizons. Under frictionless log utility, an investor who can trade every interval in a liquid market does not use those horizons: the optimal position depends only on the conditional distribution of the next return, and the investor's expected log-wealth regret from trading on a forecast is bounded by the forecast's excess expected log loss, the relative entropy D_KL(P||Q) of the truth P from the forecast Q. With transaction costs, in the quadratic model of Garleanu and Pedersen, later horizons enter through the expected-return path with weights that shift toward the next interval as costs fall. The one-step predictive is therefore the object to evaluate for that investor, its excess expected log loss controls the investor's regret without ranking imperfect forecasts' decisions, and a score pooled over horizons does not establish superiority for that decision. Two examples mark the limits: a model better at one step can be worse at twelve while iterating its own dynamics, and a forecast closer to the truth in relative entropy can induce the worse position. We recommend reporting one step ahead on its own and including an iterated one-step model among multi-step baselines.

**Keywords.** forecast evaluation; foundation models; time series; log score; Kelly criterion; myopic portfolio choice; transaction costs; direct and iterated forecasting; GIFT-Eval

**JEL codes.** C53 (forecasting and prediction methods), G11 (portfolio choice), C52 (model evaluation and selection), C44 (statistical decision theory)

**Classification / eJournals to suggest.** Econometric Modeling: Capital Markets - Forecasting; Capital Markets: Asset Pricing & Valuation; Machine Learning eJournal

**Related SSRN papers (for the "related" field).** A Stronger Univariate Benchmark for Foundational Time-Series Model Evaluation (SSRN 7583618, DOI 10.2139/ssrn.7583618).

**Disclosure line already in the paper.** Drafting assisted by Anthropic's Claude; two examples, the marginal-replacement identity and the Barron-Cover pointer from reviews by OpenAI's ChatGPT.

## Checklist before upload
- [x] Review issues on microprediction/foundationalprediction closed (#3, the Samuelson special case)
- [x] Literature check folded in (Granger-Pesaran, Pesaran-Skouras, Hakansson, Chevillon, Patton, Csiszar, Kruger-Clark-Ravazzolo; Marcellino-Stock-Watson and Campbell-Viceira wording; GIFT-Eval aggregation attributed to the leaderboard)
- [x] PDF compiles with no undefined citations or references
- [ ] Upload on SSRN and paste the abstract, keywords and JEL codes above (author action)
