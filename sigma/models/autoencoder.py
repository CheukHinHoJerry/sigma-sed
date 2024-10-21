import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channel: int,  lat_dim = 2, hidden_layer_sizes=(512, 256, 128)):

        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = building_block(self.in_channel, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], lat_dim)]

        decoder = building_block(lat_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.in_channel), nn.Softmax()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)
        return x_recon


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_channel: int,  lat_dim = 2, hidden_layer_sizes=(512, 256, 128)):

        super(VariationalAutoEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channel = in_channel
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = building_block(self.in_channel, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])

        decoder = building_block(lat_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.in_channel), nn.Softmax()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.logvar = nn.Linear(self.hls[-1], lat_dim)
        self.mu = nn.Linear(self.hls[-1], lat_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _encode(self, x):
        encoder_out = self.encoder(x)
        return self.mu(encoder_out)

    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoder_out = self.encoder(x)
        mu = self.mu(encoder_out)
        logvar = self.logvar(encoder_out)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return mu, logvar, z, x_recon

class VariationalAutoEncoder2D(nn.Module):
    def __init__(self, in_channel: int, img_size: int, lat_dim=2, hidden_layer_sizes=(32, 64, 128)):
        super(VariationalAutoEncoder2D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channel = in_channel
        self.lat_dim = lat_dim
        self.hc = hidden_layer_sizes
        self.img_size = img_size

        def conv_block(in_channel, out_channels, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02)
            )

        def deconv_block(in_channel, out_channels, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02)
            )

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(self.in_channel, self.hc[0]),
            conv_block(self.hc[0], self.hc[1]),
            conv_block(self.hc[1], self.hc[2]),
        )

        # Calculate the size after convolutional layers to define the fully connected layers
        self.fc_input_size = self.hc[-1] * (img_size // 8) * (img_size // 8)

        self.fc_mu = nn.Linear(self.fc_input_size, self.lat_dim)
        self.fc_logvar = nn.Linear(self.fc_input_size, self.lat_dim)

        # Decoder
        self.decoder_fc = nn.Linear(self.lat_dim, self.fc_input_size)

        self.decoder_conv = nn.Sequential(
            deconv_block(self.hc[2], self.hc[1]),
            deconv_block(self.hc[1], self.hc[0]),
            nn.ConvTranspose2d(self.hc[0], self.in_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use Sigmoid to get outputs between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z

    def _encode(self, x):
        batch_size = x.size(0)
        encoder_out = self.encoder_conv(x)
        encoder_out = encoder_out.view(batch_size, -1)  # Flatten
        mu = self.fc_mu(encoder_out)
        logvar = self.fc_logvar(encoder_out)
        return mu, logvar

    def _decode(self, z):
        decoder_input = self.decoder_fc(z)
        decoder_input = decoder_input.view(-1, self.hc[-1], self.img_size // 8, self.img_size // 8)
        x_recon = self.decoder_conv(decoder_input)
        return x_recon

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return mu, logvar, z, x_recon