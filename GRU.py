import torch
import torch.nn as nn

class BiGRUProteinModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_size, max_seq_len, num_layers=2, dropout=0.2):
        super(BiGRUProteinModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim+ time_emb_size, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)  # 双向GRU，所以输出维度是hidden_dim * 2
        self.max_seq_len = max_seq_len
        self.time_embedding = nn.Embedding(num_embeddings = 10, embedding_dim=time_emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein_input, time_input):
        batch_size = protein_input.size(0)
        max_protein_length =  self.max_seq_len

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(protein_input.device)
        
        time_input = time_input.repeat_interleave(max_protein_length, dim=1)
        time_input = self.time_embedding(time_input.long())

        combined_input = torch.cat((protein_input, time_input), dim=2)
        # print(combined_input.shape)

        # 双向GRU
        output, _ = self.gru(combined_input, h0)

        # 应用 dropout
        output = self.dropout(output)

        # 全连接层输出
        output = self.fc(output)

        return output
    

if __name__ == '__main__':
    # 设置一些超参数
    batch_size = 2
    max_protein_length = 10
    input_dim = 32
    hidden_dim = 64
    output_dim = 32
    time_emb_size = 16

    # 创建模型实例
    model = BiGRUProteinModel(input_dim, hidden_dim, output_dim, time_emb_size, max_protein_length)

    # 准备模拟输入数据
    protein_input = torch.randn(batch_size, max_protein_length, input_dim)
    time_input = torch.tensor([[0.5], [1.0]])

    # 将模型输入传递给forward函数
    output = model(protein_input, time_input)

    #  检查输出的形状是否符合预期
    assert output.size() == (batch_size, max_protein_length, output_dim), "Output shape is incorrect"
    print("Output shape:", output.size())
