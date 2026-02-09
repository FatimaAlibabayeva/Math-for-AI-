import numpy as np
import matplotlib.pyplot as plt
import string

# Reproducibility is key for grading
np.random.seed(42)

# ==========================================
# 1. Probability Spaces: The Birthday Paradox
# ==========================================

def theoretical_birthday_prob(k, days=365):
    """
    Computes the exact probability of at least two people sharing a birthday.
    Formula: 1 - P(all unique)
    """
    if k > days:
        return 1.0

    # TODO: Calculate P(unique)
    # Hint: (365/365) * (364/365) * ...
    # You can use np.arange or a simple loop.
    prob_unique = 1.0

    # <YOUR CODE HERE>
    for i in range(k):
        prob_unique *= (days-i)/(days)

    return 1.0 - prob_unique

def simulate_birthday_prob(k, num_trials=5000, days=365):
    """
    Estimates probability via Monte Carlo.
    """
    collisions = 0

    for _ in range(num_trials):
        # TODO: Generate k random integers in range [0, days)
        # <YOUR CODE HERE>
        birtdays = np.random.randint(0,days,size=k)
        # TODO: Check if there are duplicates
        # Hint: Comparison of len(list) vs len(set(list)) is a fast check.
        # <YOUR CODE HERE>
        if(len(set(birtdays))) < k:
            collisions+=1

    return collisions / num_trials

def run_section_1():
    print("--- Section 1: Birthday Paradox ---")
    k_values = range(1, 80) # Sufficient range to see the curve
    theory = []
    sim = []

    print("Running simulation (this may take a few seconds)...")
    for k in k_values:
        theory.append(theoretical_birthday_prob(k))
        sim.append(simulate_birthday_prob(k, num_trials=1000))

    # Plotting for Report
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, theory, label='Theoretical', linewidth=2)
    plt.plot(k_values, sim, 'o', label='Simulated', alpha=0.5, markersize=3)
    plt.axhline(0.5, color='r', linestyle='--', label='50% Threshold')
    plt.xlabel('Group Size ($k$)')
    plt.ylabel('P(Collision)')
    plt.title('Birthday Paradox: Theory vs Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 2. Conditional Probability: Spam Filter
# ==========================================

# Vocabulary and Probabilities from Assignment Table
vocab = {
    "free":    {"P_w_spam": 0.4,  "P_w_ham": 0.05},
    "prize":   {"P_w_spam": 0.2,  "P_w_ham": 0.01},
    "meeting": {"P_w_spam": 0.05, "P_w_ham": 0.5}
}

P_spam_prior = 0.3
P_ham_prior = 0.7

def classify_email(email_words):
    """
    Returns unnormalized scores for Spam and Ham.
    Score = Prior * Product(P(w|class))
    """
    # TODO: Initialize scores with Priors
    score_spam = P_spam_prior
    score_ham = P_ham_prior
    # <YOUR CODE HERE>

    for word in email_words:
        if word in vocab:
            # TODO: Multiply by P(word | class) for both classes
            # <YOUR CODE HERE>
            score_spam *= vocab[word]["P_w_spam"]
            score_ham *= vocab[word]["P_w_ham"]

    return score_spam, score_ham

def run_section_2():
    print("\n--- Section 2: Spam Classification ---")
    message = ["free", "meeting"]

    spam_score, ham_score = classify_email(message)

    print(f"Message: {message}")
    print(f"Score (Spam): {spam_score:.6f}") # Record this in Report
    print(f"Score (Ham):  {ham_score:.6f}")  # Record this in Report

    winner = "Spam" if spam_score > ham_score else "Ham"
    print(f"Prediction: {winner}")

# ==========================================
# 3. Discrete RVs: Caesar Cipher
# ==========================================

ENGLISH_CORPUS = """
probability theory is the branch of mathematics concerned with probability
although there are several different probability interpretations probability
theory treats the concept in a rigorous mathematical manner by expressing it
through a set of axioms typically these axioms formalize probability in terms
of a probability space which assigns a measure taking values between zero and
one termed the probability measure to a set of outcomes called the sample space
""".replace("\n", " ")

CIPHER_TEXT = "ZNK COXYZ YKV ZU YURBOTM ZNK VXKIOYOUT VXUHRKS OY ZU KROSOTGZK ZNK OSVUYYOHRK"

def get_pmf(text):
    """
    Computes P(Letter) for all A-Z. Returns a vector of length 26
    """
    # Helper to stick to A-Z
    text = ''.join([c.upper() for c in text if c.isalpha()])
    total = len(text)
    counts = np.zeros(26)  # burda biz Ani cixirqki Anin ASCII si bize 65 verir ve meselen Cden cixanda 2 qalir buda listeki 3cu element olur, bizede ele bu lazimdi

    # TODO: Populate counts.
    # Hint: ord('A') returns 65. Index = ord(char) - 65.
    # <YOUR CODE HERE>

    for char in text:
        index = ord(char) - 65 # ord(char) - ord('A')
        counts[index]+=1

    # Ps. ord(char)-ASCII deki qarsiqliqini qaytarir
    return counts / total if total > 0 else counts

def run_section_3():
    print("\n--- Section 3: Cryptanalysis ---")
    english_pmf = get_pmf(ENGLISH_CORPUS)
    def decrypt_caesar(cipher_text, english_pmf_ref):
        best_shift = 0
        best_decrypted_text = cipher_text # Placeholder
        return best_shift, best_decrypted_text


    shift, decrypted = decrypt_caesar(CIPHER_TEXT, english_pmf)
    decrypted_pmf = get_pmf(decrypted)

    print(f"Detected Shift: {shift}")
    print(f"Decrypted Snippet: {decrypted[:50]}...") 

    # Comparison Plot
    letters = list(string.ascii_uppercase)
    x = np.arange(len(letters))

    plt.figure(figsize=(12, 5))
    width = 0.35
    plt.bar(x - width/2, english_pmf, width, label='English Ground Truth')
    plt.bar(x + width/2, decrypted_pmf, width, label='Decrypted Text')
    plt.xticks(x, letters)
    plt.title('PMF Fingerprint Comparison')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_section_1()
    run_section_2()
    run_section_3() 