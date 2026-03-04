# Machine Learning Kaggle Competition

<div align="center">

<img src="https://img.shields.io/badge/🥈_2nd_Place-Kaggle_Competition-FFD700?style=for-the-badge&labelColor=1a1b27" />
&nbsp;
<img src="https://img.shields.io/badge/Model-Decision_Tree_Regressor-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=1a1b27" />

<br/><br/>

**[Competition Link](https://www.kaggle.com/competitions/the-virality-predictor-social-media-challenge)**

*Predicting how viral a social media video will go — using nothing but metadata, text, and a well-tuned Decision Tree.*

</div>

---

## 📌 Problem

Given a dataset of social media videos with features like author bio, video transcription, sound name, upload timestamp, follower count, region, and visual tags — predict a continuous **virality score** for unseen test videos.

---

## 🧠 Our Approach

### 1. Context-Aware Missing Data Imputation

Instead of dropping or naively filling missing values, we built **intelligent imputation functions** that leverage the textual content of each video:

- **Region Recovery** — Scanned concatenated text fields for location-specific keywords (*"tel aviv"* → `IL`, *"tokyo"* → `JP`, *"brazil"* → `BR`, etc.) and detected **Unicode character ranges** (Hebrew `\u0590-\u05FF`, Japanese/CJK `\u3040-\u9fff`) to infer the missing `user_region`.

- **Visual Tag Recovery** — Mapped missing `visual_tags` using high-confidence sound-name associations (*"Dance Monkey"* → `Bedroom_Dance`) and keyword matching (*"gym"*, *"workout"* → `Gym_Workout`).

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `text_blob` | Lowercased concatenation of bio + transcription + sound name |
| `local_hour` | Upload hour adjusted by region-specific UTC offset |
| `followers_log` | Log-transformed follower count (handles heavy skew) |
| `sound_target_enc` | Mean virality of each `sound_name` from training data |
| `emoji_count` | Non-ASCII character count in author bio |
| `bio_word_count` / `bio_len` | Word count and character length of bio |
| `trans_len` | Character length of video transcription |
| `is_weekend` | Binary flag for Saturday/Sunday uploads |

Plus **one-hot encoding** of `sound_name`, `visual_tags`, and `user_region` — aligned across train/test to prevent column mismatch.

### 3. Model

A single **Decision Tree Regressor** — no ensembles, no deep learning — with carefully tuned hyperparameters:

```python
DecisionTreeRegressor(
    max_depth=15,
    ccp_alpha=0.05,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42
)
```

All features standardized with `StandardScaler`. Performance validated with **5-fold cross-validation** before final submission.

---

## 📂 Repository Structure

```
├── README.md
└── ml.ipynb        # Full pipeline: preprocessing → feature engineering → training → submission
```

---

## 🚀 How to Run

1. Place `x_train.csv`, `y_train.csv`, and `x_test.csv` in the same directory as the notebook.
2. Open and run `ml.ipynb`.
3. A `submission.csv` file will be generated automatically.

---

## 📦 Dependencies

- Python 3.x
- `pandas` · `numpy` · `scikit-learn`

---

## 🥈 Result

**2nd place** on the final leaderboard — achieved through smart feature engineering and domain-driven imputation, proving that a well-understood simple model can outperform complex ones. 🌳
