import os

if __name__ == '__main__':

    '''
    cmd = 'python runtime.py --yaml_path options/mirnet_raw1.yaml'
    os.system(cmd)
    cmd = 'python runtime.py --yaml_path options/mprnet_raw1.yaml'
    os.system(cmd)
    cmd = 'python runtime.py --yaml_path options/mwcnn_raw1.yaml'
    os.system(cmd)
    cmd = 'python runtime.py --yaml_path options/sgn_raw1.yaml'
    os.system(cmd)
    cmd = 'python runtime.py --yaml_path options/srn_raw1.yaml'
    os.system(cmd)
    '''

    cmd = 'python validation.py --yaml_path options/mirnet_raw1.yaml'
    os.system(cmd)
    cmd = 'python validation.py --yaml_path options/mirnet_raw2.yaml'
    os.system(cmd)
    cmd = 'python validation.py --yaml_path options/mirnet_raw3.yaml'
    os.system(cmd)
