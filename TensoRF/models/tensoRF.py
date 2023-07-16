from .tensorBase import *


class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        ### 학습 될 plane, line에 대한 Tensor coefficient를 랜덤값으로 초기화함
        # self.density_n_comp = [16,4,4] -> config파일의 n_lamb_sigmar
        # self.gridSize = [141,157,94] -> aabb 크기에 맞게 재계산한 grid갯수(reso_cur) - train.py의 reconstruction() 수식 위치함
        # grid사이즈는 점점증가함
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        # print(self.density_plane)# ParameterList(
        #                         #    (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x157x141 (GPU 0)]
        #                         #    (1): Parameter containing: [torch.cuda.FloatTensor of size 1x4x94x141 (GPU 0)]
        #                         #    (2): Parameter containing: [torch.cuda.FloatTensor of size 1x4x94x157 (GPU 0)] )
        # print(self.density_line)# ParameterList(
        #                         #    (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x94x1 (GPU 0)]
        #                         #    (1): Parameter containing: [torch.cuda.FloatTensor of size 1x4x157x1 (GPU 0)]
        #                         #    (2): Parameter containing: [torch.cuda.FloatTensor of size 1x4x141x1 (GPU 0)] )
        # self.app_n_comp = [48,12,12] -> config파일의 n_lamb_sh
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)

        # self.data_dim = 27 -> opt.py 파일의 data_dim_color
        # self.basis_mat -> 입력차원 72(=48+12+12), 출력차원 27인 MLP Layer
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]  # vecMode =  [2, 1, 0]             # <- tensorBase의 init함수에 정의됨
            mat_id_0, mat_id_1 = self.matMode[i]  # matMode = [[0,1], [0,2], [1,2]]  # <- tensorBase의 init함수에 정의됨

            plane_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp, n_size),
                                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[
                                                                                                      idx]))  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):
        # print(xyz_sampled.shape)  # torch.Size([751904, 3])

        # plane + line basis
        # matMode = [[0,1], [0,2], [1,2]]  # <- tensorBase의 init함수에 정의됨
        # vecMode =  [2, 1, 0]             # <- tensorBase의 init함수에 정의됨
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)
        # print(coordinate_plane.shape)  # torch.Size([3, 751904, 1, 2]) # xy, xz, yz 좌표를 각각 추출한 배열
        # print(coordinate_line.shape)   # torch.Size([3, 751904, 1, 2]) # z, y, x    좌표를 각각 추출한 배열

        # print(self.density_plane)  # ParameterList(
        #    (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x157x141 (GPU 0)]
        #    (1): Parameter containing: [torch.cuda.FloatTensor of size 1x4x94x141 (GPU 0)]
        #    (2): Parameter containing: [torch.cuda.FloatTensor of size 1x4x94x157 (GPU 0)] )

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):  # len = 3
            ##### coordinate_plane의 좌표값을 density_plane좌표계에서 intepolation함 : xyz가 rank(=component)로 변환
            # print(coordinate_plane[idx_plane].shape) # torch.Size([751904, 1, 2])
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                             align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # print(F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).shape)
            # -> torch.Size([1, 16, 751904, 1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # print(plane_coef_point.shape)  #torch.Size([16, 751904]) -> [4, 751904] -> [4, 751904]
            # print(line_coef_point.shape)   #torch.Size([16, 751904]) -> [4, 751904] -> [4, 751904]
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        # print(sigma_feature.shape) # torch.Size([751904])
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)


'''
<<<< Example torch.nn.functional.grid_sample >>>>
Python 3.9.15
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> input = torch.arange(4*4).view(1, 1, 4, 4).float()
>>> d = torch.linspace(-1, 1, 8)
>>> print(d)
tensor([-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000])
>>> meshx, meshy = torch.meshgrid((d, d))
>>> print(meshx)
tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [-0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143],
        [-0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286],
        [-0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429],
        [ 0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429],
        [ 0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286],
        [ 0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
>>> grid = torch.stack((meshy, meshx), 2)
>>> grid = grid.unsqueeze(0) # add batch dim
>>> print(grid)
tensor([[[[-1.0000, -1.0000],
          [-0.7143, -1.0000],
          [-0.4286, -1.0000],
          [-0.1429, -1.0000],
          [ 0.1429, -1.0000],
          [ 0.4286, -1.0000],
          [ 0.7143, -1.0000],
          [ 1.0000, -1.0000]],

         [[-1.0000, -0.7143],
          [-0.7143, -0.7143],
          [-0.4286, -0.7143],
          [-0.1429, -0.7143],
          [ 0.1429, -0.7143],
          [ 0.4286, -0.7143],
          [ 0.7143, -0.7143],
          [ 1.0000, -0.7143]],

         [[-1.0000, -0.4286],
          [-0.7143, -0.4286],
          [-0.4286, -0.4286],
          [-0.1429, -0.4286],
          [ 0.1429, -0.4286],
          [ 0.4286, -0.4286],
          [ 0.7143, -0.4286],
          [ 1.0000, -0.4286]],

         [[-1.0000, -0.1429],
          [-0.7143, -0.1429],
          [-0.4286, -0.1429],
          [-0.1429, -0.1429],
          [ 0.1429, -0.1429],
          [ 0.4286, -0.1429],
          [ 0.7143, -0.1429],
          [ 1.0000, -0.1429]],

         [[-1.0000,  0.1429],
          [-0.7143,  0.1429],
          [-0.4286,  0.1429],
          [-0.1429,  0.1429],
          [ 0.1429,  0.1429],
          [ 0.4286,  0.1429],
          [ 0.7143,  0.1429],
          [ 1.0000,  0.1429]],

         [[-1.0000,  0.4286],
          [-0.7143,  0.4286],
          [-0.4286,  0.4286],
          [-0.1429,  0.4286],
          [ 0.1429,  0.4286],
          [ 0.4286,  0.4286],
          [ 0.7143,  0.4286],
          [ 1.0000,  0.4286]],

         [[-1.0000,  0.7143],
          [-0.7143,  0.7143],
          [-0.4286,  0.7143],
          [-0.1429,  0.7143],
          [ 0.1429,  0.7143],
          [ 0.4286,  0.7143],
          [ 0.7143,  0.7143],
          [ 1.0000,  0.7143]],

         [[-1.0000,  1.0000],
          [-0.7143,  1.0000],
          [-0.4286,  1.0000],
          [-0.1429,  1.0000],
          [ 0.1429,  1.0000],
          [ 0.4286,  1.0000],
          [ 0.7143,  1.0000],
          [ 1.0000,  1.0000]]]])
>>> print(grid.shape)
torch.Size([1, 8, 8, 2])
>>> print(meshx)
tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [-0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143, -0.7143],
        [-0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286, -0.4286],
        [-0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429, -0.1429],
        [ 0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429,  0.1429],
        [ 0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286,  0.4286],
        [ 0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143,  0.7143],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
>>> print(meshy)
tensor([[-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000],
        [-1.0000, -0.7143, -0.4286, -0.1429,  0.1429,  0.4286,  0.7143,  1.0000]])
>>> print(input)
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
>>> print(input.shape)
torch.Size([1, 1, 4, 4])
    plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
>>> output = torch.nn.functional.grid_sample(input, grid)
>>> print(output)
tensor([[[[ 0.0000,  0.0357,  0.3214,  0.6071,  0.8929,  1.1786,  1.4643, 0.7500],
          [ 0.1429,  0.3571,  0.9286,  1.5000,  2.0714,  2.6429,  3.2143, 1.6429],
          [ 1.2857,  2.6429,  3.2143,  3.7857,  4.3571,  4.9286,  5.5000, 2.7857],
          [ 2.4286,  4.9286,  5.5000,  6.0714,  6.6429,  7.2143,  7.7857, 3.9286],
          [ 3.5714,  7.2143,  7.7857,  8.3571,  8.9286,  9.5000, 10.0714, 5.0714],
          [ 4.7143,  9.5000, 10.0714, 10.6429, 11.2143, 11.7857, 12.3571, 6.2143],
          [ 5.8571, 11.7857, 12.3571, 12.9286, 13.5000, 14.0714, 14.6429, 7.3571],
          [ 3.0000,  6.0357,  6.3214,  6.6071,  6.8929,  7.1786,  7.4643, 3.7500]]]])
>>> print(output.shape)
torch.Size([1, 1, 8, 8])

'''