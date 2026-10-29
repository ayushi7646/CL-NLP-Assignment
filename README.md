# CL & NLP assignment: LSTM practice questions

Name: Ayushi Solanki  
SAP ID: 500119389  
Course: Computational Linguistics & NLP (CSEG4134)

## About LSTM

An LSTM (Long Short-Term Memory) network is a type of recurrent neural network built to handle long sequences. A plain RNN struggles to remember things from many steps back because its gradients shrink during backpropagation through time. LSTMs fix this by adding a cell state, a separate memory line that runs through the sequence and is changed only through gates.

Each LSTM cell has three gates plus a candidate value:

- The forget gate decides how much of the old cell state to keep.
- The input gate decides how much new information to write.
- The candidate value is the new information itself, computed with tanh.
- The output gate decides how much of the cell state goes into the hidden state.

The cell state update is C_t = f_t · C_(t−1) + i_t · C̃_t and the hidden state is h_t = o_t · tanh(C_t). Because the cell state is updated by addition, the gradient can pass back through many time steps without dying out, as long as the forget gate stays close to 1. That is why LSTMs work well for text, speech and time-series forecasting.

## Questions covered

1. Forward pass through a single scalar LSTM cell
2. Counting parameters in an LSTM layer (one bias and two bias conventions)
3. Vanishing gradient through the cell state
4. Electricity load forecasting with a one-week lookback window
5. Predicting pump failure from vibration data

## Answer summary

| Q | Answer |
|---|--------|
| 1 | f_t = 0.6248, i_t = 0.5939, C̃_t = 0.4542, C_t = 0.5196, o_t = 0.6106, h_t = 0.2915 |
| 2 | (a) 5504 (b) 5632 |
| 3 | 0.9^10 = 0.3487, 0.9^50 = 0.0052, 0.5^50 ≈ 8.88 × 10⁻¹⁶, 0.99^50 = 0.6050 |
| 4 | 26280 records, 26089 windows, input (64, 168, 3) and output (64, 24), 18205 training windows in 285 batches, 17408 parameters |
| 5 | 0.75, predicts failure, TP 30, FP 15, FN 10, TN 945, precision 0.6667, recall 0.75, F1 0.7059, accuracy 0.975 |
