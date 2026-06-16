"""Generate a larger, harder RAG eval corpus + matching gold set.

For each fact we emit:
  - the ANSWER paragraph (contains the unique gold keyword)
  - a near-duplicate DISTRACTOR paragraph (semantically close, same vocabulary,
    but WITHOUT the answer/keyword) — this is what makes pure-vector retrieval
    occasionally rank the wrong passage first.
Plus many topical FILLER paragraphs to scale the corpus to hundreds of chunks.

The gold set is derived directly from the fact bank, so labels are correct by
construction. Deterministic (fixed seed) for reproducibility.

Run:  python eval/build_corpus.py   then   python eval/evaluate_rag.py
"""
import json
import os
import random

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(EVAL_DIR, "corpus")
GOLD_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")
FILLER_PER_TOPIC = 34
SEED = 7

# (topic, keyword, answer_sentence, paraphrased_question, near-duplicate distractor)
FACTS = [
    # databases
    ("databases", "superkey", "In BCNF, every functional dependency X to Y requires that X is a superkey of the relation.", "What must the left side of every dependency be under Boyce-Codd Normal Form?", "BCNF is a normal form used in relational schema design to reduce update anomalies."),
    ("databases", "transitive", "Third normal form eliminates transitive dependencies among non-key attributes.", "Which normal form removes indirect dependencies between non-key columns?", "Third normal form is a common target when designing relational tables."),
    ("databases", "write-ahead", "Durability is guaranteed by write-ahead logging, which records changes to the log before the data pages.", "How does a database survive a crash without losing committed data?", "Logging is essential for database recovery and for auditing changes over time."),
    ("databases", "dirty read", "A dirty read happens when a transaction reads data written by another uncommitted transaction.", "What anomaly lets one transaction see another's uncommitted writes?", "Isolation levels govern how concurrent transactions interact with each other."),
    ("databases", "hash index", "A hash index gives constant-time point lookups but cannot serve range queries.", "Which index structure supports equality lookups but not ranges?", "Indexes speed up query execution at the cost of extra storage and write overhead."),
    ("databases", "covering index", "A covering index contains every column a query needs so no table row lookup is required.", "What kind of index lets a query be answered from the index alone?", "Composite indexes combine several columns to support particular query patterns."),

    # algorithms
    ("algorithms", "E log V", "Dijkstra's algorithm runs in O(E log V) time when implemented with a binary-heap priority queue.", "What is the running-time complexity of Dijkstra's shortest-path method?", "Dijkstra's algorithm finds shortest paths from a single source using a priority queue."),
    ("algorithms", "negative", "Bellman-Ford handles graphs with negative edge weights and can detect negative cycles.", "Which shortest-path algorithm works when some edges have negative cost?", "Bellman-Ford repeatedly relaxes all edges to compute shortest paths."),
    ("algorithms", "memoization", "Dynamic programming avoids recomputation by storing subproblem results, a technique called memoization.", "What is the name for caching subproblem answers to speed up recursion?", "Dynamic programming breaks a problem into overlapping subproblems."),
    ("algorithms", "pivot", "Quicksort degrades to quadratic time when a poor pivot is chosen on already-sorted input.", "Under what condition does quicksort hit its worst-case running time?", "Quicksort is a divide-and-conquer sorting algorithm with good average performance."),
    ("algorithms", "depth-first", "Depth-first search explores each branch fully using a stack before backtracking.", "Which traversal goes as deep as possible before backing up?", "Graph traversal visits the nodes and edges in some systematic order."),
    ("algorithms", "level by level", "Breadth-first search visits nodes level by level using a FIFO queue and finds unweighted shortest paths.", "Which search expands a graph one layer at a time?", "Search algorithms differ in the order in which they visit graph vertices."),

    # networking
    ("networking", "three-way", "TCP sets up a connection with a three-way handshake of SYN, SYN-ACK, and ACK.", "How does TCP establish a new connection between two hosts?", "TCP is a reliable, connection-oriented transport-layer protocol."),
    ("networking", "congestion", "TCP congestion control uses slow start and congestion avoidance to adjust the send rate.", "How does TCP avoid overwhelming the network path?", "TCP provides reliable delivery using sequence numbers and acknowledgments."),
    ("networking", "connectionless", "UDP is connectionless and unreliable but has low overhead, suiting streaming and DNS.", "Which transport protocol sends datagrams without establishing a session?", "UDP operates at the transport layer directly above IP."),
    ("networking", "128", "An IPv6 address is 128 bits long, far larger than IPv4's 32-bit address.", "How many bits make up an IPv6 address?", "IPv6 was designed to replace the exhausted IPv4 address space."),
    ("networking", "rewriting addresses", "NAT lets many private hosts share a single public IP address by rewriting addresses.", "What technique allows multiple internal devices to use one public address?", "Private address ranges are reserved for use inside local networks."),
    ("networking", "stateless", "HTTP is a stateless request-response protocol, so cookies are used to track sessions.", "Why does the web rely on cookies given how HTTP works?", "HTTP is the application-layer protocol that underlies the web."),

    # operating_systems
    ("operating_systems", "circular wait", "Deadlock requires four conditions at once: mutual exclusion, hold and wait, no preemption, and circular wait.", "What conditions must all hold simultaneously for a deadlock?", "Deadlock is a state in which threads block one another indefinitely."),
    ("operating_systems", "mutual exclusion", "A mutex enforces mutual exclusion so only one thread enters a critical section at a time.", "Which primitive guarantees exclusive access to a critical section?", "Synchronization primitives coordinate access to shared data structures."),
    ("operating_systems", "not resident", "A page fault occurs when a referenced page is not resident and must be loaded from disk.", "What event fires when a process touches memory not currently in RAM?", "Paging divides memory into fixed-size blocks managed by the operating system."),
    ("operating_systems", "lookaside", "The translation lookaside buffer caches recent virtual-to-physical address translations.", "What hardware cache speeds up address translation?", "Virtual memory maps each process address to a physical frame."),
    ("operating_systems", "thrashing", "Thrashing happens when excessive paging leaves almost no time for useful work.", "What is it called when a system spends most of its time swapping pages?", "Memory pressure can severely degrade overall system performance."),
    ("operating_systems", "time quantum", "Round-robin scheduling gives each ready process a fixed time quantum in rotation.", "Which scheduler hands every process an equal slice of CPU in turn?", "CPU schedulers decide which ready process should run next."),

    # machine_learning
    ("machine_learning", "memorizes", "Overfitting is when a model memorizes training noise and generalizes poorly to new data.", "What problem occurs when a model fits the training set too closely?", "Model evaluation uses a held-out test set to estimate generalization."),
    ("machine_learning", "penalizes", "L2 regularization penalizes large weights to reduce overfitting.", "Which technique adds a penalty on weight magnitude to improve generalization?", "Hyperparameters control how a learning algorithm behaves during training."),
    ("machine_learning", "opposite the loss", "Gradient descent updates parameters in the direction opposite the loss gradient.", "What optimization method steps downhill along the loss surface?", "Training a model means minimizing a loss function over the data."),
    ("machine_learning", "predicted positives", "Precision is the fraction of predicted positives that are actually correct.", "Which metric measures how many positive predictions were right?", "Classification quality can be summarized with several complementary metrics."),
    ("machine_learning", "actual positives", "Recall is the fraction of actual positives that the model correctly identified.", "Which metric captures how many of the true positives were found?", "A confusion matrix summarizes the outcomes of a classifier."),
    ("machine_learning", "dense vectors", "An embedding maps discrete tokens into dense vectors that capture semantic similarity.", "What maps words into continuous vectors that encode meaning?", "Vectors are compared with cosine similarity inside retrieval systems."),

    # security
    ("security", "rainbow tables", "A password salt is random data added before hashing to defeat precomputed rainbow tables.", "What random value stops attackers from using precomputed hash tables?", "Password hashes should be produced with slow algorithms such as bcrypt."),
    ("security", "arbitrary queries", "SQL injection lets an attacker run arbitrary queries by smuggling SQL through unsanitized input.", "What attack abuses unsanitized input to execute database commands?", "Input validation is a core defensive programming practice."),
    ("security", "victim's browser", "Cross-site scripting injects malicious scripts that run in a victim's browser.", "Which web attack runs attacker scripts in another user's browser?", "Web applications must escape their output to stay safe."),
    ("security", "same key", "Symmetric encryption uses the same key to encrypt and decrypt, as in AES.", "Which encryption type shares one key for both directions?", "Encryption protects the confidentiality of data in transit and at rest."),
    ("security", "key pair", "Asymmetric cryptography uses a public key to encrypt and a private key to decrypt, forming a key pair.", "Which scheme uses a pair of keys where one is shared openly?", "Key management is a major challenge in cryptographic systems."),
    ("security", "authenticated requests", "A CSRF attack tricks a logged-in user's browser into sending unwanted authenticated requests.", "Which attack forces a victim's browser to make state-changing requests?", "Tokens and same-site cookies help protect authenticated web sessions."),
]

FILLER_TEMPLATES = [
    "In {t}, practitioners often revisit core principles when designing real systems, case {i}.",
    "A study group reviewed several {t} topics this week, focusing on practical examples, note {i}.",
    "Common {t} pitfalls appear in interviews and should be understood conceptually, item {i}.",
    "The {t} chapter includes worked exercises and diagrams to build intuition, section {i}.",
    "Engineers compare trade-offs between approaches in {t} before committing to a design, memo {i}.",
    "Lecture {i} on {t} emphasized definitions, assumptions, and typical failure modes.",
    "A summary sheet collects key {t} terms with short explanations for quick review, page {i}.",
    "Discussion {i} explored how {t} ideas connect to systems students have already built.",
]


def _topic_label(topic: str) -> str:
    return topic.replace("_", " ")


def build():
    rng = random.Random(SEED)
    os.makedirs(CORPUS_DIR, exist_ok=True)
    # wipe old corpus
    for f in os.listdir(CORPUS_DIR):
        os.remove(os.path.join(CORPUS_DIR, f))

    topics = sorted({f[0] for f in FACTS})
    gold = []
    total_paras = 0

    for topic in topics:
        paras = []
        for (_, kw, ans, q, dist) in [f for f in FACTS if f[0] == topic]:
            paras.append(ans)
            paras.append(dist)
            gold.append({"question": q, "source": f"{topic}.md", "keyword": kw})
        for i in range(FILLER_PER_TOPIC):
            tmpl = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
            paras.append(tmpl.format(t=_topic_label(topic), i=i))
        rng.shuffle(paras)
        total_paras += len(paras)
        with open(os.path.join(CORPUS_DIR, f"{topic}.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# {_topic_label(topic).title()}\n\n")
            fh.write("\n\n".join(paras) + "\n")

    with open(GOLD_PATH, "w", encoding="utf-8") as fh:
        for g in gold:
            fh.write(json.dumps(g) + "\n")

    print(f"Wrote {len(topics)} files, {total_paras} paragraphs, {len(gold)} gold questions.")


if __name__ == "__main__":
    build()
