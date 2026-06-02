class regModel(nn.Module):
    def __init__(self, pro_dim=480, drug_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()

        self.proj_pro = nn.Sequential(
            nn.Linear(pro_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.proj_drug = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )

        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.attention_weights = None

    def forward(self, pro, drug):
        pro = self.proj_pro(pro).unsqueeze(1)
        drug = self.proj_drug(drug).unsqueeze(1)

        drug_to_protein, attn1 = self.cross_attn(drug, pro, pro)
        protein_to_drug, attn2 = self.cross_attn(pro, drug, drug)

        self.attention_weights = {'drug_to_protein': attn1, 'protein_to_drug': attn2}

        interaction = torch.cat([drug_to_protein.squeeze(1),
                                 protein_to_drug.squeeze(1)], dim=1)

        interaction_feat = self.interaction_net(interaction)
        out = self.regressor(interaction_feat)
        return out.squeeze()

class InteractionModel(nn.Module):
    def __init__(self, pro_dim=480, drug_dim=384, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.protein_proj = nn.Sequential(
            nn.Linear(pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, pro, drug):
        pro_emb = self.protein_proj(pro)
        drug_emb = self.drug_proj(drug)
        pro_emb = F.normalize(pro_emb, p=2, dim=1)
        drug_emb = F.normalize(drug_emb, p=2, dim=1)
        dot_product = (pro_emb * drug_emb).sum(dim=1, keepdim=True)
        temp = self.temperature.abs() + 1e-8
        logit = dot_product / temp + self.bias
        prob = torch.sigmoid(logit)
        return prob.squeeze(dim=1)
