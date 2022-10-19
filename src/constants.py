import torch 

# define device
CUDA = 'cuda:2'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
SEED = 1234

# NTA YTA keywords
NTA_KEYWORDS = ['nta', 'nah', 'you are not the asshole', 'you\'re not the asshole', 'u are not the asshole', 'u re not the asshole', 
                'you re not the asshole', 'u\'re not the asshole', 'not the asshole', 'not the ah', 'not asshole', 'not ah']
YTA_KEYWORDS = ['yta', 'you are the asshole', 'you\'re the asshole', 'u are the asshole', 'u re the asshole', 
                'you re the asshole', 'u\'re the asshole', 'you the ah', 'you the asshole', 'u the asshole', 'u the ah']

DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"