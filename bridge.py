import torch
import torch.nn as nn

class Bridge(nn.Module):
    
    def __init__(self, hidden_size, bidirectional):
        super(Bridge, self).__init__()
        self._bidirectional = bidirectional
        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self._hidden_projection = nn.Linear(encoder_output_size, hidden_size)
        self._cell_projection = nn.Linear(encoder_output_size, hidden_size)
        self._encoder_output_projection = nn.Linear(encoder_output_size, hidden_size)

    def forward(self, encoder_output, final_encoder_states):
        # encoder_output: Tensor (batch_size, time_step, hidden_size * num_directions)
        # final_encoder_hidden: Tensor (num_layers * num_directions, batch_size, hidden_size)
        # final_encoder_cell: Tensor (num_layers * num_directions, batch_size, hidden_size)
        final_encoder_hidden, final_encoder_cell = final_encoder_states
        if self._bidirectional:
            final_encoder_hidden = torch.cat(final_encoder_hidden.chunk(chunks=2, dim=0), dim=2)
            final_encoder_cell = torch.cat(final_encoder_cell.chunk(chunks=2, dim=0), dim=2)
        init_decoder_hidden = torch.stack([self._hidden_projection(h) for h in final_encoder_hidden], dim=0)
        init_decoder_cell = torch.stack([self._cell_projection(c) for c in final_encoder_cell], dim=0)
        encoder_memory = self._encoder_output_projection(encoder_output)
        init_decoder_states = (init_decoder_hidden, init_decoder_cell)
        return encoder_memory, init_decoder_states