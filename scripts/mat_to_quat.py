# convert matrix to quaternion
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np

def main(path):
    # load data from pkl
    with open(path, 'rb') as f:
        data = pickle.load(f)
    trans_matrix = data['base_T_cam']

    # use scipy to convert rot_matrix to quaternion
    r = R.from_matrix(trans_matrix[:3, :3])
    quat = r.as_quat()

    # displacement
    disp = trans_matrix[:3, -1]
    
    # print
    print(f'  qx: {quat[0]} \n  qy: {quat[1]} \n  qz: {quat[2]} \n  qw: {quat[3]}')
    print(f'  x: {disp[0]} \n  y: {disp[1]} \n  z: {disp[2]}')

    print(f'{disp[0]} {disp[1]} {disp[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}')

    # inverse matrix
    trans_matrix_inv = np.linalg.inv(data['base_T_cam'])
    r_inv = R.from_matrix(trans_matrix_inv[:3, :3])
    quat_inv = r_inv.as_quat()
    disp_inv = trans_matrix_inv[:3, -1]

    print(f'  qx: {quat_inv[0]} \n  qy: {quat_inv[1]} \n  qz: {quat_inv[2]} \n  qw: {quat_inv[3]}')
    print(f'  x: {disp_inv[0]} \n  y: {disp_inv[1]} \n  z: {disp_inv[2]}')

    print(f'{disp_inv[0]} {disp_inv[1]} {disp_inv[2]} {quat_inv[0]} {quat_inv[1]} {quat_inv[2]} {quat_inv[3]}')

    print(data)

if __name__ == '__main__':
    main('/home/sqz/projects/fairo/polymetis/calib_handeye.pkl')
