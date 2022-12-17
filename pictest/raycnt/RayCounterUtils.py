import cv2
import torch
import numpy as np
import open3d as o3d
from svox2 import SparseGrid, Rays

class RayCounter:
    '''
    statistic grid visit
    '''

    counts: torch.Tensor
    grid: SparseGrid

    def __init__(self, grid: SparseGrid) -> None:
        '''
        @param grid: SparseGrid
        '''
        self.grid = grid
        self.counts = torch.zeros_like(grid.links)

    def countAndSave(self, path, rays: Rays, addToTotal: bool=True, countGrid: torch.Tensor=None, rayLoss: torch.Tensor=None, iters: int=1, eps=1e-8, spreadMeanRayLoss=True, extraInfo: dict=None):
        '''
        @param path: save path (should be better if end with .npy)
        @param rays: Rays (Origins + Dirs)
        @param addToTotal: if True, the result is add to self.counts
        @param countGrid: replace self.counts by countGrid (just in this function)
        @param rayLoss:   spread ray loss evenly to its passing grid (without distance weight) [B]
        @param iters: rayLoss spread iters (just leave it as default)
        @param eps: a very small number to prevent divide by 0
        @param spreadMeanRayLoss: if True, spread (ray loss / passed grid cnt) to grid, else spread (ray loss) directly
        @param extraInfo: extra info that you want to save in .npy file (will OVERWRITE the output result if have key confliction)

        @returns {
            'cnt': current ray grid visit [H x W x D]
            'total': total ray grid visit [H x W x D]
            'ray': True, 
            'origins': ray origins (after transform)       [B x 3]
            'dirs': ray direction vector (after transform) [B x 3]
            'tmax': ray max sample point distance (after transform) [B]
            'raysum': ray grid visit sum   [B]
            'raycnt': ray grid visit count [B]
            'raymean': ray grid visit mean [B]
            'rayloss': ray loss that spread to grid [H x W x D]
            
            [keys that in @param{extraInfo.keys()}]: extraInfo[ key ]
        }
        '''
        
        saveDict = self.count(rays=rays, addToTotal=addToTotal, countGrid=countGrid, rayLoss=rayLoss, iters=iters, eps=eps, spreadMeanRayLoss=spreadMeanRayLoss)
        
        if extraInfo is None:
            extraInfo = dict()
        for key in extraInfo.keys():
            saveDict[key] = extraInfo[key]

        return self.saveNpy(path=path, saveDict=saveDict)

    def count(self, rays: Rays, addToTotal: bool=True, countGrid: torch.Tensor=None, rayLoss: torch.Tensor=None, iters: int=1, eps=1e-8, spreadMeanRayLoss=True) -> dict:
        '''
        @param rays: Rays (Origins + Dirs)
        @param addToTotal: if True, the result is add to self.counts
        @param countGrid: replace self.counts by countGrid (just in this function)
        @param rayLoss:   spread ray loss evenly to its passing grid (without distance weight) [B]
        @param iters: rayLoss spread iters (just leave it as default)
        @param eps: a very small number to prevent divide by 0
        @param spreadMeanRayLoss: if True, spread (ray loss / passed grid cnt) to grid, else spread (ray loss) directly

        @returns {
            'cnt': current ray grid visit [H x W x D]
            'total': total ray grid visit [H x W x D]
            'ray': True, 
            'origins': ray origins (after transform)       [B x 3]
            'dirs': ray direction vector (after transform) [B x 3]
            'tmax': ray max sample point distance (after transform) [B]
            'raysum': ray grid visit sum   [B]
            'raycnt': ray grid visit count [B]
            'raymean': ray grid visit mean [B]
            'rayloss': ray loss that spread to grid [H x W x D]
        }
        '''
        
        countResult = self._count(rays=rays, addToTotal=addToTotal, countGrid=countGrid, eps=eps)

        if rayLoss is None:
            return countResult

        assert iters is not None and iters > 0, 'iters should > 0 when rayLoss is not None'

        rayLossCnt = torch.zeros_like(self.grid.links)

        for iter in range(iters):
            deltaRayLossCnt = self.countRayLoss(countResult=countResult, rayLoss=rayLoss, rayLossCnt=rayLossCnt, eps=eps, spreadMeanRayLoss=spreadMeanRayLoss)
            rayLossCnt = rayLossCnt + deltaRayLossCnt
        
        countResult['rayloss'] = rayLossCnt / iters
        return countResult

    def _count(self, rays: Rays, addToTotal, countGrid, eps):
        origins = self.grid.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self.grid._grid_size()
        dirs = dirs * (self.grid._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        invdirs = 1.0 / dirs

        gsz = self.grid._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        gsz = gsz_cu
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.grid.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        tMaxResult: torch.Tensor = tmax

        summResult = torch.zeros(B, device=origins.device)
        good_indices = torch.arange(B, device=origins.device)
        currentVisCnt = torch.zeros(B, device=origins.device)

        dirResult = dirs
        originResult = origins

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        summ = 0

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        tmax = tmax[mask]

        counts = None
        if countGrid is not None:
            counts = countGrid.detach().clone().float()
        else:
            counts = self.counts.detach().clone().float()
        counts.requires_grad = True

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)

            links000 = counts[lx, ly, lz]
            links001 = counts[lx, ly, lz + 1]
            links010 = counts[lx, ly + 1, lz]
            links011 = counts[lx, ly + 1, lz + 1]
            links100 = counts[lx + 1, ly, lz]
            links101 = counts[lx + 1, ly, lz + 1]
            links110 = counts[lx + 1, ly + 1, lz]
            links111 = counts[lx + 1, ly + 1, lz + 1]

            tmpCount = links000 + links001 + links010 + links011 + links100 + links101 + links110 + links111

            summ = summ + tmpCount.sum()
            summResult[good_indices] = summResult[good_indices] + tmpCount
            currentVisCnt[good_indices] = currentVisCnt[good_indices] + 1

            t += self.grid.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            tmax = tmax[mask]

        deltaCount = torch.autograd.grad(summ, counts)[0]

        if addToTotal:
            self.counts = self.counts + deltaCount
        
        return {
            'cnt': deltaCount.detach(), 
            'total': self.counts.detach(), 
            'ray': True, 
            'origins': originResult.detach(), 
            'dirs': dirResult.detach(), 
            'tmax': tMaxResult.detach(), 
            'raysum': summResult.detach(), 
            'raycnt': currentVisCnt.detach(), 
            'raymean': (summResult / (currentVisCnt + eps)).detach()
        }

    def countRayLoss(self, countResult, rayLoss: torch.Tensor, rayLossCnt: torch.Tensor, eps, spreadMeanRayLoss: bool):
        origins = countResult['origins']
        dirs = countResult['dirs']
        rayCounts = countResult['raycnt']
        invdirs = 1.0 / dirs
        B = dirs.size(0)

        gsz = self.grid._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        gsz = gsz_cu
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs
        del invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.grid.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values

        good_indices = torch.arange(B, device=origins.device)

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        t = t[mask]
        tmax = tmax[mask]
        rayCounts = rayCounts[mask]
        rayLoss = rayLoss[mask]

        rayLossCountTemp = rayLossCnt.detach().clone().float()
        rayLossCountTemp.requires_grad = True
        rayLossSumm = 0

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)

            sigma000 = rayLossCountTemp[lx, ly, lz]
            sigma001 = rayLossCountTemp[lx, ly, lz + 1]
            sigma010 = rayLossCountTemp[lx, ly + 1, lz]
            sigma011 = rayLossCountTemp[lx, ly + 1, lz + 1]
            sigma100 = rayLossCountTemp[lx + 1, ly, lz]
            sigma101 = rayLossCountTemp[lx + 1, ly, lz + 1]
            sigma110 = rayLossCountTemp[lx + 1, ly + 1, lz]
            sigma111 = rayLossCountTemp[lx + 1, ly + 1, lz + 1]

            wa, wb = 1.0 - pos, pos

            c00 = sigma000 * wa[:, 2] + sigma001 * wb[:, 2]
            c01 = sigma010 * wa[:, 2] + sigma011 * wb[:, 2]
            c10 = sigma100 * wa[:, 2] + sigma101 * wb[:, 2]
            c11 = sigma110 * wa[:, 2] + sigma111 * wb[:, 2]
            c0 = c00 * wa[:, 1] + c01 * wb[:, 1]
            c1 = c10 * wa[:, 1] + c11 * wb[:, 1]
            sigma = c0 * wa[:, 0] + c1 * wb[:, 0]

            if spreadMeanRayLoss:
                sigma = sigma * rayLoss.detach() / (rayCounts + eps)
            else:
                sigma = sigma * rayLoss.detach()
            rayLossSumm = rayLossSumm + sigma.sum()

            t += self.grid.opt.step_size

            mask = (t <= tmax)
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            tmax = tmax[mask]
            rayCounts = rayCounts[mask]
            rayLoss = rayLoss[mask]
        
        deltaRayLossCnt = torch.autograd.grad(rayLossSumm, rayLossCountTemp)[0]
        return deltaRayLossCnt.detach()

    def saveNpy(self, path, saveDict: dict) -> dict:
        for key in saveDict.keys():
            if isinstance(saveDict[key], torch.Tensor):
                saveDict[key] = saveDict[key].detach().cpu().numpy()

        np.save(path, saveDict)
        return saveDict

class RayVisualizer:
    def visualization(self, path, showTotal: bool=True, showRay: bool=False, eps=1e-10):
        saveDict = np.load(path, allow_pickle=True).item()

        showData = None
        if showTotal:
            showData = saveDict['total']
        else:
            showData = saveDict['cnt']

        minn = np.min(showData)
        maxx = np.max(showData)

        showData = (showData - minn) / (maxx - minn + eps) * 255

        H, W, D = showData.shape
        showData = np.reshape(showData, (1, -1))

        showData = showData.astype(np.uint8)
        showData = cv2.applyColorMap(showData, cv2.COLORMAP_RAINBOW).astype(np.float32)[0] / 255

        X = np.arange(H)
        Y = np.arange(W)
        Z = np.arange(D)
        X, Y, Z = np.meshgrid(X, Y, Z)
        points = np.reshape(np.stack((X, Y, Z), axis=-1), (-1, 3))

        ply_point_cloud = o3d.geometry.PointCloud()
        ply_point_cloud.points = o3d.utility.Vector3dVector(points)
        ply_point_cloud.colors = o3d.utility.Vector3dVector(showData)

        o3d.visualization.draw_geometries([ply_point_cloud])

        if saveDict['ray'] and showRay:
            pass
        else:
            pass