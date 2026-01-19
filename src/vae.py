import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        
    def encode(self, x):
        raise NotImplementedError
        
    def decode(self, z):
        raise NotImplementedError
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        raise NotImplementedError

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(), # Swish
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.silu(x + self.block(x))

class ResNetVAE(BaseVAE):
    def __init__(self, input_channels=1, latent_dim=10, hidden_dims=[32, 64, 128, 256]):
        super(ResNetVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResidualBlock(h_dim)
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flatten size (assuming 64x64 input -> 32->16->8->4)
        self.flatten_size = hidden_dims[-1] * 4 * 4
        
        self.fc_mean = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.SiLU(),
                    ResidualBlock(hidden_dims[i+1])
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1),
            # nn.Sigmoid() # Assuming data is [0, 1], enable if needed
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mean(h), self.fc_logvar(h)
        
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4) # Hardcoded for now based on hidden_dims[-1]=256
        h = self.decoder(h)
        return self.final_layer(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class VAE(BaseVAE):
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=10):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
        self.act = nn.SiLU()
        
    def encode(self, x):
        h = self.act(self.encoder_fc1(x))
        return self.encoder_fc2_mean(h), self.encoder_fc2_logvar(h)
    
    def decode(self, z):
        h = self.act(self.decoder_fc1(z))
        return self.decoder_fc2(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class ConvVAE(BaseVAE):
    def __init__(self, input_channels=1, latent_dim=10):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten()
        )
        
        self.fc_mean = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)
        
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class HybridVAE(BaseVAE):
    def __init__(self, audio_dim=128, lyrics_dim=100, hidden_dim=64, latent_dim=10):
        super(HybridVAE, self).__init__()
        
        self.act = nn.SiLU()
        
        # Audio Encoder
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            self.act
        )
        
        # Lyrics Encoder
        self.lyrics_enc = nn.Sequential(
            nn.Linear(lyrics_dim, hidden_dim),
            self.act
        )
        
        # Fusion
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoders
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * 2)
        
        self.audio_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, audio_dim)
        )
        
        self.lyrics_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, lyrics_dim)
        )
        
    def encode(self, audio, lyrics):
        h_audio = self.audio_enc(audio)
        h_lyrics = self.lyrics_enc(lyrics)
        h = torch.cat([h_audio, h_lyrics], dim=1)
        return self.fc_mean(h), self.fc_logvar(h)
        
    def decode(self, z):
        h = self.decoder_input(z)
        h_audio, h_lyrics = torch.chunk(h, 2, dim=1)
        return self.audio_dec(h_audio), self.lyrics_dec(h_lyrics)
        
    def forward(self, x):
        audio, lyrics = x
        mu, logvar = self.encode(audio, lyrics)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_lyrics = self.decode(z)
        return (recon_audio, recon_lyrics), mu, logvar

class CVAE(BaseVAE):
    def __init__(self, input_dim=128, n_classes=5, hidden_dim=64, latent_dim=10):
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.act = nn.SiLU()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim + n_classes, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x, y):
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float().to(x.device)
        inputs = torch.cat([x, y_onehot], dim=1)
        h = self.act(self.encoder_fc1(inputs))
        return self.encoder_fc2_mean(h), self.encoder_fc2_logvar(h)
    
    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float().to(z.device)
        inputs = torch.cat([z, y_onehot], dim=1)
        h = self.act(self.decoder_fc1(inputs))
        return self.decoder_fc2(h)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction term
    if isinstance(recon_x, tuple) and isinstance(x, tuple):
        # Hybrid case
        MSE = 0
        for rx, tx in zip(recon_x, x):
            MSE += F.mse_loss(rx, tx, reduction='sum')
    else:
        MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + beta * KLD
