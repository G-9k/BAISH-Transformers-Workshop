import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLanguageModel(nn.Module):
    """
    MLP-based language model with fixed context window.
    Uses learned embeddings and hidden layers (unlike n-gram lookup tables).
    """
    def __init__(self, vocab_size, context_length, embed_dim, hidden_dim):
        super().__init__()
        self.context_length = context_length
        
        # TODO: Create embedding layer
        # Maps each token to a learned vector of size embed_dim
        # Hint: nn.Embedding(vocab_size, embed_dim)
        self.embeddingLayer = nn.Embedding(vocab_size, embed_dim)
        
        # TODO: Create first hidden layer
        # Input size: context_length * embed_dim (flattened embeddings)
        # Output size: hidden_dim
        # Hint: nn.Linear(in_features, out_features)
        inputSize = context_length * embed_dim
        outputSize = hidden_dim
        self.hiddenLayer1 = nn.Linear(inputSize, outputSize)
        
        # TODO: Create second hidden layer
        # Input size: hidden_dim
        # Output size: hidden_dim
        self.hiddenLayer2 = nn.Linear(hidden_dim, hidden_dim)

        self.hiddenLayer3 = nn.Linear(hidden_dim, hidden_dim)
        
        # TODO: Create output layer
        # Input size: hidden_dim
        # Output size: vocab_size (prediction for each possible token)
        self.outputLayer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, idx):
        """
        Args:
            idx: (batch, context_length) tensor of token indices
        Returns:
            logits: (batch, vocab_size)
        """
        # TODO: Get embeddings for context tokens
        # Shape should be: (batch, context_length, embed_dim)
        embeddings = self.embeddingLayer(idx)
        
        # TODO: Flatten embeddings
        # Shape should be: (batch, context_length * embed_dim)
        # Hint: Use .view(batch_size, -1)
        embeddings = embeddings.view(idx.shape[0], -1)
        
        # TODO: Pass through first hidden layer + activation
        # Hint: Use F.relu or torch.relu for activation
        h1 = self.hiddenLayer1(embeddings)
        h1 = F.relu(h1)
        
        # TODO: Pass through second hidden layer + activation
        h2 = self.hiddenLayer2(h1)
        h2 = F.relu(h2)
        
        h3 = self.hiddenLayer2(h2)
        h3 = F.relu(h3)
        # TODO: Pass through output layer (no activation - raw logits)
        # Return shape: (batch, vocab_size)
        res = self.outputLayer(h3)

        return res
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        for _ in range(max_new_tokens):
            # Crop context to last context_length tokens
            idx_cond = idx[:, -self.context_length:]
            
            # Pad if needed (for start of sequence)
            if idx_cond.shape[1] < self.context_length:
                padding = torch.zeros(idx.shape[0], self.context_length - idx_cond.shape[1], 
                                     dtype=torch.long)
                idx_cond = torch.cat([padding, idx_cond], dim=1)
            
            # Get predictions
            logits = self(idx_cond)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def get_batch(data, context_length, batch_size):
    """
    Sample random batches from data.
    Returns context windows and targets.
    """
    # TODO: Sample random starting indices
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    # TODO: Extract context windows
    x = torch.stack([data[i:i+context_length] for i in ix])
    
    # TODO: Get target tokens (next token after each context)
    y = data[ix + context_length]
    
    return x, y


def estimate_loss(model, data, context_length, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size)
        # TODO: Get model predictions
        logits = model.forward(X)
        
        # TODO: Calculate loss
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 16      # Look at 8 previous characters
    embed_dim = 48          # Size of token embeddings
    hidden_dim = 128        # Size of hidden layers
    
    # Load and encode data
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    print(f"Context length: {context_length}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Hidden dim: {hidden_dim}")
    
    # TODO: Create model
    model = MLPLanguageModel(vocab_size, context_length, embed_dim, hidden_dim)
    
    # TODO: Print parameter count
    # Hint: sum(p.numel() for p in model.parameters())
    print(sum(p.numel() for p in model.parameters()))
    
    # TODO: Create optimizer (Adam is good for MLPs)
    # Hint: torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nTraining...")
    for iter in range(max_iters):
        xb, yb = get_batch(train_data, context_length, batch_size)
        
        # TODO: Forward pass
        logits = model.forward(xb)  # (batch, vocab_size)
        
        # TODO: Calculate loss
        loss = F.cross_entropy(logits, yb)
        
        # TODO: Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, context_length, batch_size)
            val_loss = estimate_loss(model, val_data, context_length, batch_size)
            print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))