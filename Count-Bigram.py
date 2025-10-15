import torch
import torch.nn.functional as F

class CountBasedBigram:
    """
    Count-based bigram model using PyTorch tensors.
    Builds a frequency table, no gradient descent needed!
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # TODO: Create a (vocab_size, vocab_size) tensor to store counts
        # Hint: Use torch.zeros
        # counts[i, j] = how many times token i is followed by token j
        self.counts = torch.ones((vocab_size, vocab_size))
    
    def train(self, data):
        """
        Count bigram frequencies in the data.
        Args:
            data: torch.LongTensor of token indices
        """
        # TODO: Iterate through consecutive pairs in data
        # For each pair (current_token, next_token):
        #   Increment counts[current_token, next_token]
        # Hint: Use data[:-1] for current tokens, data[1:] for next tokens
        # Hint: You can use a for loop with zip, or vectorized operations
        for current, next in zip(data[:-1], data[1:]):
            self.counts[current, next] = self.counts[current, next] + 1
    
    def get_probabilities(self):
        """
        Convert counts to probabilities.
        Returns: (vocab_size, vocab_size) tensor where [i, j] = P(j | i)
        """
        # TODO: Normalize counts by row sums to get probabilities
        # P(next | current) = count(current, next) / sum(count(current, *))
        # Hint: Use self.counts.sum(dim=1, keepdim=True) to get row sums
        # Hint: Add small epsilon to avoid division by zero
        sum = self.counts.sum(dim=1, keepdim=True)
        shape = self.counts.shape
        probs = torch.zeros(shape)

        probs = self.counts / sum

        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         probs[i, j] = self.counts[i, j] / (sum + 0.00001)

        return probs
    
    def generate(self, idx, max_new_tokens):
        """
        Generate tokens by sampling from the count-based distribution.
        Args:
            idx: (batch, time) starting context
            max_new_tokens: number of tokens to generate
        """
        probs = self.get_probabilities()
        
        for _ in range(max_new_tokens):
            # TODO: Get the last token in the sequence
            # Hint: Use idx[:, -1]
            current_token = idx[:, -1]
            
            # TODO: Get probability distribution for next token
            # Hint: Use probs[current_token] to get the row(s)
            next_probs = probs[current_token]
            
            # TODO: Sample from the distribution
            # Hint: Use torch.multinomial(next_probs, num_samples=1)
            idx_next = torch.multinomial(next_probs, num_samples=1)
            
            # TODO: Append to sequence
            # Hint: Use torch.cat((idx, idx_next), dim=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def calculate_loss(self, data):
        """
        Calculate cross-entropy loss (negative log-likelihood).
        Same metric as neural models!
        """
        probs = self.get_probabilities()
        
        # TODO: Get current and next tokens
        current_tokens = data[:-1]  # data[:-1]
        next_tokens = data[1:]     # data[1:]
        
        # TODO: Get probabilities for the actual next tokens that occurred
        # Hint: Use probs[current_tokens, next_tokens] for fancy indexing
        token_probs = probs[current_tokens, next_tokens]
        
        # TODO: Calculate negative log likelihood
        # Hint: Use torch.log() and handle zeros with clamp or add small epsilon
        # Return the mean



        # for i, row in enumerate(token_probs):
        #     for j, elem in enumerate(token_probs[row]):
        #         if elem == 0: token_probs[i][j] = 0.0001
    
        CE = -torch.log(token_probs).mean()
        return CE


if __name__ == "__main__":
    # Load and encode data
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    #Tokenize data
    chars = set(text)
    vocab_size = len(chars)
    stoi = {char: i for i, char in enumerate(chars)}
    itos = {value: key for key, value in stoi.items()}
    data = torch.tensor([stoi[char] for char in text]) 
    
    # Split data
    n = int(0.8 * len(data)) 
    train_data = data[n:]
    val_data = data[:n]
    #
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    
    # TODO: Create model
    model = CountBasedBigram(vocab_size)
    
    # TODO: Train (just count!)
    print("\nCounting bigrams...")
    model.train(train_data)
    print("Done!")
    
    # TODO: Calculate losses
    train_loss = model.calculate_loss(train_data[:10000])
    val_loss = model.calculate_loss(val_data[:10000])
    #
    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, 500)
    print(''.join([itos[number] for number in generated[0].tolist()]))