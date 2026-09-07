from typing import List, Dict, Any

QUESTION_BANK: List[Dict[str, Any]] = [
    {
        "id": 1,
        "category": "Software Engineering",
        "subcategory": "Algorithms & Data Structures",
        "difficulty": "Medium",
        "company": "Google / Meta",
        "title": "Two Sum & Hash Map Trade-offs",
        "question": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`. Explain the time and space complexity of your approach vs the brute force method.",
        "hint": "Think about using a Hash Map to store the complement (`target - num`) as you iterate through the array in O(n) time.",
        "model_answer": "Use a hash map to map array values to their indices. Iterate through the array once: for each number x, check if (target - x) exists in the map. If so, return [map[target - x], current_index]. Time Complexity: O(n), Space Complexity: O(n)."
    },
    {
        "id": 2,
        "category": "Software Engineering",
        "subcategory": "Algorithms & Data Structures",
        "difficulty": "Hard",
        "company": "Amazon / Apple",
        "title": "LRU Cache Implementation",
        "question": "Design and implement a Least Recently Used (LRU) Cache data structure. It should support `get(key)` and `put(key, value)` operations in average O(1) time complexity.",
        "hint": "Combine a Doubly Linked List for O(1) node insertion/removal with a Hash Map for O(1) key-to-node lookups.",
        "model_answer": "Use a Hash Map pointing to nodes in a Doubly Linked List. When an element is accessed or added, move its node to the head of the list. When capacity is exceeded, remove the node at the tail (least recently used)."
    },
    {
        "id": 3,
        "category": "System Design",
        "subcategory": "Distributed Systems",
        "difficulty": "Hard",
        "company": "Uber / Netflix",
        "title": "Design a Distributed URL Shortener (e.g. TinyURL)",
        "question": "Design a high-throughput, low-latency URL shortener service. Address unique hash generation, database storage choice, caching strategy, and custom alias collisions.",
        "hint": "Consider Base62 encoding of an auto-incrementing ID or KGS (Key Generation Service) pre-generating 7-character hashes.",
        "model_answer": "Key components: 1) API Gateway with rate limiting. 2) Key Generation Service (KGS) pre-generating Base62 strings. 3) NoSQL DB (e.g., Cassandra or DynamoDB) for key-value mapping. 4) Redis cache layer for popular short URLs (80/20 rule). 5) Redirection with 301 (Permanent) vs 302 (Temporary) depending on analytics needs."
    },
    {
        "id": 4,
        "category": "System Design",
        "subcategory": "Scalability & Databases",
        "difficulty": "Medium",
        "company": "Meta / Stripe",
        "title": "SQL vs NoSQL & Database Sharding",
        "question": "Explain when you would choose a relational database (PostgreSQL) vs a document store (MongoDB). How would you horizontal shard a high-write database?",
        "hint": "Evaluate ACID compliance, relational integrity, schema flexibility, and read/write access patterns.",
        "model_answer": "SQL is preferred for complex queries, strict ACID constraints, and structured schema (financial/user auth). NoSQL fits unstructured/semi-structured data with rapid iteration and high write throughput. Sharding involves partitioning data based on a Shard Key (e.g. hash(user_id) % N) with a directory lookup or consistent hashing."
    },
    {
        "id": 5,
        "category": "Data Science & ML",
        "subcategory": "Machine Learning",
        "difficulty": "Medium",
        "company": "OpenAI / Microsoft",
        "title": "Overfitting, Underfitting & Regularization",
        "question": "How do you detect overfitting in a machine learning model? Compare L1 (Lasso) and L2 (Ridge) regularization mechanisms.",
        "hint": "Observe the divergence between training loss and validation loss over epochs.",
        "model_answer": "Overfitting occurs when training error is low but test/validation error is high. L1 regularization adds |w| (absolute weights) promoting sparsity (feature selection). L2 adds w^2 (squared weights) which shrinks coefficients smoothly toward zero."
    },
    {
        "id": 6,
        "category": "Product Management",
        "subcategory": "Product Strategy",
        "difficulty": "Medium",
        "company": "Google / Airbnb",
        "title": "Metrics for Launching a New Feature",
        "question": "How would you define success metrics for introducing an AI assistant into an e-commerce mobile application?",
        "hint": "Categorize metrics into North Star Metric, Engagement Metrics, Conversion Rate, and Counter Metrics.",
        "model_answer": "North Star Metric: Increase in overall checkout conversion rate. Engagement Metrics: Daily Active Users (DAU) using AI chat, query depth. Conversion: Add-to-cart rate from AI recommendations. Counter Metrics: Customer support ticket volume, latency, bounce rate."
    },
    {
        "id": 7,
        "category": "Behavioral",
        "subcategory": "STAR Method",
        "difficulty": "Easy",
        "company": "Amazon / Microsoft",
        "title": "Handling Conflict with Teammates",
        "question": "Describe a situation where you had a strong technical disagreement with a colleague or tech lead. How did you handle it and what was the outcome?",
        "hint": "Structure your answer using STAR: Situation, Task, Action, Result. Focus on objective data, benchmarking, and collaboration.",
        "model_answer": "Situation: Disagreed on database migration strategy under tight deadline. Task: Resolve architectural deadlock without delaying launch. Action: Built quick proof-of-concept benchmark measuring latency & error rates for both approaches. Result: Agreed on data-driven winner, launched on time with zero downtime."
    },
    {
        "id": 8,
        "category": "Behavioral",
        "subcategory": "Leadership & Failure",
        "difficulty": "Medium",
        "company": "Amazon / Meta",
        "title": "Managing Production Incidents & Failures",
        "question": "Tell me about a time when a system you deployed caused a failure or outage in production. How did you remediate it and what were your key takeaways?",
        "hint": "Highlight ownership, fast rollback/mitigation, root cause analysis (RCA), and preventative measures.",
        "model_answer": "Situation: A missing null check caused memory leaks in API pod. Task: Restore service immediately. Action: Triggered automated canary rollback, updated alert thresholds, and wrote regression test suite. Result: Service restored in 4 minutes, post-mortem created 3 long-term guardrails."
    }
]

def get_filtered_questions(category="All", difficulty="All", search_query=""):
    filtered = QUESTION_BANK
    if category != "All":
        filtered = [q for q in filtered if q["category"] == category]
    if difficulty != "All":
        filtered = [q for q in filtered if q["difficulty"] == difficulty]
    if search_query.strip():
        q_lower = search_query.lower()
        filtered = [
            q for q in filtered
            if q_lower in q["title"].lower() or q_lower in q["question"].lower() or q_lower in q["subcategory"].lower()
        ]
    return filtered
