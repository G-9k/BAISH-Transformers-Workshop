import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralBigram(nn.Module):
    """
    Neural network version of bigram model.
    Same as count-based, but learns the probability table via gradient descent.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # TODO: Create embedding table (vocab_size, vocab_size)
        # This learns the same thing as the count table, but via optimization
        self.vocabSize = vocab_size
        self.embeddingTable = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx):
        """
        Args:
            idx: (batch,) or (batch, 1) tensor of current token indices
        Returns:
            logits: (batch, vocab_size) predictions for next token
        """
        # TODO: Handle both (batch,) and (batch, 1) shapes
        if idx.dim() == 1:
            idx = idx.unsqueeze(-1) # agregamos una dimensión al final

        # TODO: Pass idx through embedding table to get logits
        logits = self.embeddingTable(idx).squeeze(1)
        return logits
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        for _ in range(max_new_tokens):
            # TODO: Get last token
            current = idx[:, -1:]
            
            # TODO: Get predictions
            logits = self(current)
            
            # TODO: Apply softmax to get probabilities
            probs = torch.softmax(logits,-1)
            
            # TODO: Sample next token
            idx_next = torch.multinomial(probs, 1)
            
            # TODO: Append to sequence
            idx = torch.cat((idx, idx_next), 1)
        
        return idx


def get_batch(data, batch_size):
    """
    Sample random batches from data.
    Returns context (bigram uses 1 token) and targets.
    """
    # TODO: Sample random indices (not too close to end)
    ix = torch.randint(0, len(data)-10, (batch_size,))
    
    # TODO: Get current tokens as context
    x = data[ix]
    
    # TODO: Get next tokens as targets
    y = data[ix+1]
    
    return x, y


def estimate_loss(model, data, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size)
        # TODO: Get model predictions
        logits = model.forward(X)

        
        # TODO: Calculate cross-entropy loss
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    max_iters = 10000
    eval_interval = 500
    learning_rate = 3.0
    
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
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    
    # TODO: Create model
    model = NeuralBigram(vocab_size)
    
    # TODO: Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        # TODO: Get batch
        xb, yb = get_batch(data, batch_size)
        
        # TODO: Forward pass
        logits = model.forward(xb) 
        
        # TODO: Calculate loss
        loss = F.cross_entropy(logits, yb)
        
        # TODO: Backward pass
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, batch_size)
            val_loss = estimate_loss(model, val_data, batch_size)
            print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))