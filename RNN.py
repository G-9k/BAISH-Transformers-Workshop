import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLanguageModel(nn.Module):
    """
    Vanilla RNN language model.
    Processes sequences recurrently with hidden state.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # TODO: Create embedding layer
        # Hint: nn.Embedding(vocab_size, embed_dim)
        self.embeddingLayer = nn.Embedding(vocab_size, embed_dim)
        
        # TODO: Create RNN layer
        # Hint: nn.RNN(input_size, hidden_size, batch_first=True)
        # batch_first=True means input/output shapes are (batch, seq, feature)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        
        # TODO: Create output layer
        # Hint: nn.Linear(hidden_size, vocab_size)
        self.outputLayer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, idx, hidden=None):
        """
        Args:
            idx: (batch, seq_len) tensor of token indices
            hidden: (1, batch, hidden_size) initial hidden state (optional)
        Returns:
            logits: (batch, vocab_size) prediction for next token
            hidden: (1, batch, hidden_size) final hidden state
        """
        # TODO: Get embeddings
        # Shape: (batch, seq_len, embed_dim)
        embeddings = self.embeddingLayer(idx)
        
        # TODO: Pass through RNN
        # Hint: output, hidden = self.rnn(embeddings, hidden)
        # output shape: (batch, seq_len, hidden_size)
        # hidden shape: (1, batch, hidden_size)
        output, hiddLayer = self.rnn(embeddings, hidden)
        
        # TODO: Take last timestep output
        # Hint: output[:, -1, :]
        # Shape: (batch, hidden_size)
        lto = output[:, -1, :]
        
        # TODO: Pass through output layer
        # Shape: (batch, vocab_size)
        logits = self.outputLayer(lto)
        
        return logits, hiddLayer
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        hidden = None  # Start with no hidden state
        
        for _ in range(max_new_tokens):
            # TODO: Get predictions (pass hidden state!)
            logits, hidden = self(idx, hidden)  # self(idx, hidden)
            
            # TODO: Convert to probabilities
            probs = F.softmax(logits, -1)
            
            # TODO: Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # TODO: Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Same data loading functions as MLP
def get_batch(data, context_length, batch_size):
    """Sample random batches from data."""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = data[ix + context_length]
    return x, y


def estimate_loss(model, data, context_length, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size)
        # TODO: Get predictions (no need to keep hidden state during eval)
        logits, _ = model(X)
        
        # TODO: Calculate loss
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    steps_per_epoch = 1000
    eval_interval = 2
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 64        # RNNs can handle longer sequences
    embed_dim = 32
    hidden_size = 128          # RNN hidden state size
    
    # ... rest of training code same as MLP ...

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
    print(f"Hidden dim: {hidden_size}")
    
    # TODO: Create model
    model = RNNLanguageModel(vocab_size, embed_dim, hidden_size)
    
    # TODO: Print parameter count
    # Hint: sum(p.numel() for p in model.parameters())
    print(sum(p.numel() for p in model.parameters()))
    
    # TODO: Create optimizer (Adam is good for MLPs)
    # Hint: torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step in range(steps_per_epoch):
            xb, yb = get_batch(train_data, context_length, batch_size)

            # Forward pass
            logits, _ = model(xb)

            # Calculate loss
            loss = F.cross_entropy(logits, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate loss

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            train_loss = estimate_loss(model, train_data, context_length, batch_size)
            val_loss = estimate_loss(model, val_data, context_length, batch_size)
            print(f"epoch {epoch + 1}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 1000)[0].tolist()))