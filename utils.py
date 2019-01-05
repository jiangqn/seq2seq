INIT = 1e-2

def reorder_sequence(sequence_embedding, order):
    # sequence_embedding: Tensor (batch_size, time_step, embed_size)
    # order: list (batch_size,)
    assert sequence_embedding.size(0) == len(order)
    order = torch.LongTensor(order).cuda()
    return sequence_embedding.index_select(index=order, dim=0)

def reorder_lstm_states(states, order):
    # states: (hidden, cell)
    # hidden: Tensor (num_layers * num_directions, batch_size, hidden_size)
    # cell: Tensor (num_layers * num_directions, batch_size, hidden_size)
    assert isinstance(states, tuple)
    assert len(states) == 2
    assert states[0].size(1) == len(order)
    order = torch.LongTensor(order).cuda()
    return (
        states[0].index_select(index=order, dim=1),
        states[1].index_select(index=order, dim=1)
    )