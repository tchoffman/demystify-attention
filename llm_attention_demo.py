"""
Demystifying the Attention Layer in LLMs
PyAtl Presentation - February 2024

This notebook demonstrates:
1. Tokenization and vectorization
2. Word embeddings and vector arithmetic
3. Attention mechanisms
4. How context modifies meaning
"""

# ============================================================================
# SETUP AND INSTALLATIONS
# ============================================================================

# Install required packages (uncomment if needed)
# !pip install transformers torch numpy matplotlib seaborn gensim bertviz

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries loaded successfully!")

# ============================================================================
# PART 1: TOKENIZATION - Breaking Text into Pieces
# ============================================================================

print("\n" + "="*70)
print("PART 1: TOKENIZATION")
print("="*70)

# Load a tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def demonstrate_tokenization(text):
    """Show how text gets broken into tokens"""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"\nOriginal text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(tokens)}")
    return tokens, token_ids

# Pre-loaded examples
example_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is fascinating!",
    "PyAtl is awesome!"
]

print("\n--- Pre-loaded Tokenization Examples ---")
for text in example_texts:
    demonstrate_tokenization(text)

# Interactive section
print("\n--- Try Your Own! ---")
print("Use: demonstrate_tokenization('your text here')")

# ============================================================================
# PART 2: WORD EMBEDDINGS - Words as Vectors
# ============================================================================

print("\n" + "="*70)
print("PART 2: WORD EMBEDDINGS AND VECTOR ARITHMETIC")
print("="*70)

# Load pre-trained word embeddings (using BERT for simplicity)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

def get_word_embedding(word):
    """Get the embedding vector for a single word"""
    # Tokenize and get embedding
    inputs = bert_tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the [CLS] token embedding or average
    embedding = outputs.last_hidden_state[0, 1, :].numpy()  # First real token
    return embedding

def vector_arithmetic_demo(word1, operation, word2, operation2, word3, top_n=5):
    """
    Demonstrate vector arithmetic: word1 - word2 + word3
    Example: king - man + woman ≈ queen
    """
    print(f"\n🧮 Computing: {word1} {operation} {word2} {operation2} {word3}")
    
    # Get embeddings
    v1 = get_word_embedding(word1)
    v2 = get_word_embedding(word2)
    v3 = get_word_embedding(word3)
    
    # Perform arithmetic
    if operation == '-' and operation2 == '+':
        result_vector = v1 - v2 + v3
    else:
        print("Currently only supports 'word1 - word2 + word3' format")
        return
    
    # Test against a vocabulary of common words
    test_words = [
        'queen', 'woman', 'girl', 'princess', 'king', 'man', 'boy', 'prince',
        'italy', 'france', 'spain', 'germany', 'mussolini', 'napoleon', 'franco',
        'jordan', 'jackson', 'tyson', 'phelps', 'basketball', 'music', 'boxing',
        'mother', 'father', 'sister', 'brother', 'aunt', 'uncle',
        'paris', 'london', 'rome', 'berlin', 'madrid'
    ]
    
    # Calculate similarities
    similarities = []
    for word in test_words:
        if word.lower() not in [word1.lower(), word2.lower(), word3.lower()]:
            emb = get_word_embedding(word)
            similarity = 1 - cosine(result_vector, emb)
            similarities.append((word, similarity))
    
    # Sort and display top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n📊 Top {top_n} most similar words:")
    for i, (word, sim) in enumerate(similarities[:top_n], 1):
        print(f"  {i}. {word:15s} (similarity: {sim:.4f})")
    
    return similarities

# Pre-loaded examples
print("\n--- Classic Example: Gender Analogy ---")
vector_arithmetic_demo('king', '-', 'man', '+', 'woman')

print("\n--- Historical Leaders Example ---")
vector_arithmetic_demo('hitler', '-', 'germany', '+', 'italy')

print("\n--- Sports Example ---")
vector_arithmetic_demo('jordan', '-', 'basketball', '+', 'music')

print("\n--- Try your own! ---")
print("Use: vector_arithmetic_demo('word1', '-', 'word2', '+', 'word3')")

# ============================================================================
# PART 3: ATTENTION MECHANISM - The Magic Happens Here
# ============================================================================

print("\n" + "="*70)
print("PART 3: ATTENTION MECHANISM")
print("="*70)

def visualize_attention(text, layer=0, head=0):
    """
    Visualize attention weights for a given text
    Shows which words the model pays attention to
    """
    # Tokenize
    inputs = bert_tokenizer(text, return_tensors='pt')
    
    # Get attention weights
    with torch.no_grad():
        outputs = bert_model(**inputs, output_attentions=True)
    
    # Extract attention for specified layer and head
    attention = outputs.attentions[layer][0, head].numpy()
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    plt.title(f'Attention Weights - Layer {layer}, Head {head}\n"{text}"')
    plt.xlabel('Key (attending to)')
    plt.ylabel('Query (attending from)')
    plt.tight_layout()
    plt.show()
    
    return attention, tokens

# Pre-loaded examples showing how attention captures context
attention_examples = [
    "Michael Jordan plays basketball",
    "Michael Jackson made music",
    "The bank by the river",
    "Money in the bank",
]

print("\n--- Attention Pattern Examples ---")
print("These show which words the model focuses on when processing each word")
print("\nExecute: visualize_attention('your sentence here')")

# ============================================================================
# PART 4: CONTEXTUAL EMBEDDINGS - Same Word, Different Meanings
# ============================================================================

print("\n" + "="*70)
print("PART 4: CONTEXTUAL EMBEDDINGS")
print("="*70)

def compare_contextualized_embeddings(word, sentence1, sentence2):
    """
    Show how the same word gets different embeddings in different contexts
    This is the KEY insight about attention!
    """
    print(f"\n🎯 Analyzing the word '{word}' in different contexts:")
    print(f"  Context 1: '{sentence1}'")
    print(f"  Context 2: '{sentence2}'")
    
    # Get embeddings for word in both contexts
    def get_contextual_embedding(word, sentence):
        inputs = bert_tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # Find the position of the target word
        tokens = bert_tokenizer.tokenize(sentence)
        word_tokens = bert_tokenizer.tokenize(word)
        
        # Find where our word appears (simple matching)
        for i, token in enumerate(tokens):
            if word.lower() in token.lower():
                # +1 because of [CLS] token
                embedding = outputs.last_hidden_state[0, i+1, :].numpy()
                return embedding
        
        return None
    
    emb1 = get_contextual_embedding(word, sentence1)
    emb2 = get_contextual_embedding(word, sentence2)
    
    if emb1 is not None and emb2 is not None:
        # Calculate cosine similarity
        similarity = 1 - cosine(emb1, emb2)
        print(f"\n📐 Cosine similarity between the two embeddings: {similarity:.4f}")
        print(f"   (1.0 = identical, 0.0 = completely different)")
        
        if similarity > 0.9:
            print("   → Very similar meanings in both contexts")
        elif similarity > 0.7:
            print("   → Somewhat similar meanings")
        else:
            print("   → Quite different meanings! Attention modified the representation.")
    
    return emb1, emb2, similarity

# Pre-loaded examples
print("\n--- Example 1: Polysemous Word 'Bank' ---")
compare_contextualized_embeddings(
    'bank',
    'I deposited money at the bank',
    'We sat on the river bank'
)

print("\n--- Example 2: Name 'Michael' with Different People ---")
compare_contextualized_embeddings(
    'Michael',
    'Michael Jordan won six NBA championships',
    'Michael Jackson was the king of pop'
)

print("\n--- Example 3: Word 'Apple' ---")
compare_contextualized_embeddings(
    'apple',
    'I ate a delicious apple',
    'Apple released a new iPhone'
)

print("\n--- Try your own! ---")
print("Use: compare_contextualized_embeddings('word', 'sentence 1', 'sentence 2')")

# ============================================================================
# PART 5: NEXT TOKEN PREDICTION - It's More Than It Seems!
# ============================================================================

print("\n" + "="*70)
print("PART 5: NEXT TOKEN PREDICTION")
print("="*70)

# Load GPT-2 for generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.eval()

def predict_next_tokens(prompt, top_k=10):
    """
    Show the top-k most likely next tokens
    This demonstrates that 'predicting the next word' requires
    understanding context, syntax, semantics, and reasoning
    """
    print(f"\n🔮 Predicting next token after: '{prompt}'")
    
    # Tokenize input
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    
    # Get predictions
    with torch.no_grad():
        outputs = gpt2_model(inputs)
        predictions = outputs.logits
    
    # Get probabilities for next token
    next_token_logits = predictions[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=0)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(next_token_probs, top_k)
    
    print(f"\n📊 Top {top_k} most likely next tokens:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token = gpt2_tokenizer.decode([idx])
        print(f"  {i}. '{token}' (probability: {prob:.4f})")
    
    return top_probs, top_indices

# Pre-loaded examples
print("\n--- Example 1: Simple Completion ---")
predict_next_tokens("The capital of France is")

print("\n--- Example 2: Context-Dependent ---")
predict_next_tokens("After winning the NBA championship, Michael Jordan")

print("\n--- Example 3: Requires World Knowledge ---")
predict_next_tokens("Python is a programming")

print("\n--- Try your own! ---")
print("Use: predict_next_tokens('your prompt here')")

# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)

print("""
1. 🔤 TOKENIZATION: Text is broken into tokens (subwords) for processing

2. 📊 EMBEDDINGS: Words are represented as vectors in high-dimensional space
   - Similar concepts are closer together
   - Vector arithmetic captures semantic relationships

3. 🎯 ATTENTION: The mechanism that allows context to modify meaning
   - Same word, different contexts → different internal representations
   - The model learns which words to pay attention to

4. 🧠 LAYERS: Deeper layers capture more abstract/complex patterns
   - Early layers: syntax and simple patterns
   - Later layers: reasoning, world knowledge, complex relationships

5. 💡 "JUST PREDICTING THE NEXT WORD":
   Yes, but to do so accurately requires:
   - Understanding grammar and syntax
   - Modeling semantic relationships
   - Incorporating world knowledge
   - Reasoning about context
   
   Saying an LLM "just predicts the next word" is like saying a 
   chess grandmaster "just moves pieces" - technically true but 
   misses the sophisticated internal modeling that makes it possible!
""")

print("\n✨ Thanks for exploring LLMs with me! Questions?")
print("\n" + "="*70)