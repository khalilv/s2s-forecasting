import torch

def main():
    filename = '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/no_rel_humid/6h_finetune/checkpoints/last.ckpt'
    ckpt = torch.load(filename, map_location=torch.device('cpu'))

    rel_humid_idxs = [34,35,36,37,38,39,40]
    non_rel_humid_idxs = [i for i in range(48) if i not in rel_humid_idxs]
    ckpt['state_dict']['net.channel_embed'] = ckpt['state_dict']['net.channel_embed'][:, non_rel_humid_idxs]
    ckpt['state_dict']['net.head.4.weight'] = torch.cat([ckpt['state_dict']['net.head.4.weight'][:int(34*4)], ckpt['state_dict']['net.head.4.weight'][41*4:]], dim=0)
    ckpt['state_dict']['net.head.4.bias'] = torch.cat([ckpt['state_dict']['net.head.4.bias'][:int(34*4)], ckpt['state_dict']['net.head.4.bias'][41*4:]], dim=0)
    for idx in rel_humid_idxs:
        ckpt['state_dict'][f'net.token_embeds.{idx}.proj.weight'] = ckpt['state_dict'][f'net.token_embeds.{idx + 7}.proj.weight']
        ckpt['state_dict'][f'net.token_embeds.{idx}.proj.bias'] = ckpt['state_dict'][f'net.token_embeds.{idx + 7}.proj.bias']
        del ckpt['state_dict'][f'net.token_embeds.{idx + 7}.proj.weight']
        del ckpt['state_dict'][f'net.token_embeds.{idx + 7}.proj.bias']
    
    updated_vars = ['land_sea_mask', 'orography', 'latitude', '2m_temperature', '10m_u_component_of_wind', 
                '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500',
                'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',
                'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500',
                'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850',
                'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250',
                'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700',
                'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50',
                'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700',
                'temperature_850', 'temperature_925', 'specific_humidity_50',
                'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600',
                'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925']
    
    # Create dictionary mapping variables to indices
    var_to_idx = {var: idx for idx, var in enumerate(updated_vars)}
    ckpt['var_to_idx'] = var_to_idx
    # Save the updated checkpoint back to the same file
    torch.save(ckpt, '/glade/derecho/scratch/kvirji/s2s-forecasting/pretrained_weights/climaX-5.625-cmip6-no-rel-humidity.ckpt')
    return ckpt

if __name__ == '__main__':
    model_state = main()
