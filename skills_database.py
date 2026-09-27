"""
skills_database.py
-------------------
A hand-curated lexicon of skills, grouped by category. This is what
lets us do lexicon-based skill extraction (no ML needed): we just check
which of these phrases appear in the lemmatized resume/JD text.

Feel free to extend these lists for your own domain.
"""

SKILLS_DB = {
    "programming_languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
        "sql", "bash", "shell scripting", "perl", "dart",
    ],
    "web_development": [
        "html", "css", "react", "angular", "vue", "node.js", "express",
        "django", "flask", "fastapi", "spring boot", "next.js", "redux",
        "rest api", "graphql", "webpack", "tailwind",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "keras", "data analysis", "data visualization", "statistics",
        "regression", "classification", "clustering", "feature engineering",
        "tf-idf", "cosine similarity", "lemmatization", "tokenization",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite",
        "cassandra", "dynamodb", "elasticsearch", "firebase",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
        "terraform", "ansible", "git", "github", "gitlab", "linux",
        "microservices", "lambda", "cloudformation",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "problem-solving", "time management", "project management",
        "critical thinking", "adaptability", "collaboration", "mentoring",
        "presentation", "negotiation", "stakeholder management",
    ],
    "tools": [
        "jira", "confluence", "figma", "excel", "power bi", "tableau",
        "postman", "slack", "notion", "trello",
    ],
}


def get_all_skills():
    """Flat set of every skill phrase across all categories."""
    all_skills = set()
    for skills in SKILLS_DB.values():
        all_skills.update(skills)
    return all_skills


def get_skill_category(skill: str):
    for category, skills in SKILLS_DB.items():
        if skill in skills:
            return category
    return "other"
