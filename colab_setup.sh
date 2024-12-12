#!/usr/bin/env bash

# Not provided by default in Google Colab
pip install jaxtyping

# The rest downloads some binary files we'll need
pip install gdown
# Python pickle with training events
gdown --id 1oyecHzwWVgYX2unTsV45kfltE7Jg85sg
# Initial parameters for active network
gdown --id 1P_Ke-XEnnr_gSdSjjm7SHhROeepgu-ww
# Initial parameters for target network
gdown --id 1OybDPtnMA7wI5V0MS5SQG3GnMOCj03jB
