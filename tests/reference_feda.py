"""Original (loop-based) FEDA implementation from ID-Blau/PrepareCondition, kept verbatim as a reference."""
import torch


def compute_v0_a(flow_fwd, flow_bwd):
    return 0.5 * (flow_fwd - flow_bwd), 0.5 * (flow_fwd + flow_bwd)


def deviation_accumulation_scatter_add_v3(events_list, flow_fwd_list, flow_bwd_list, grid_size, alpha=1.0, device='cuda'):
    H, W = grid_size
    R_flat = torch.zeros(H * W, device=device, dtype=torch.float32)
    N_flat = torch.zeros(H * W, device=device, dtype=torch.float32)
    x_list, y_list, t_list, p_list = events_list
    if len(flow_fwd_list) != len(flow_bwd_list):
        flow_bwd_list.insert(0, None)
    for i in range(len(x_list)):
        x = x_list[i].long().to(device)
        y = y_list[i].long().to(device)
        t = t_list[i].float().to(device)
        p = p_list[i].float().to(device)
        dt = (t - t[0]) / (t[-1] - t[0] + 1e-12)
        if flow_bwd_list[i] is None:
            quad_flow = flow_fwd_list[i].to(device)
            qf0 = quad_flow[0, 0, y, x]
            qf1 = quad_flow[0, 1, y, x]
            flow_mags = torch.sqrt((qf0 * dt) ** 2 + (qf1 * dt) ** 2)
        else:
            v0_map, a_map = compute_v0_a(flow_fwd_list[i].to(device), flow_bwd_list[i].to(device))
            v0_events = v0_map[0, :, y, x].permute(1, 0)
            a_events = a_map[0, :, y, x].permute(1, 0)
            dt_expanded = dt.view(-1, 1)
            integrated_flow = v0_events * dt_expanded + a_events * (dt_expanded ** 2)
            flow_mags = torch.sqrt(integrated_flow[:, 0] ** 2 + integrated_flow[:, 1] ** 2)
        weight = 1.0 + alpha * flow_mags
        d = p * weight
        pixel_idx = y * W + x
        sorted_pixel_idx, sort_idx = torch.sort(pixel_idx)
        sorted_d = d[sort_idx]
        unique_pix, counts = torch.unique_consecutive(sorted_pixel_idx, return_counts=True)
        orders = torch.cat([torch.arange(c, device=device) for c in counts])
        group_counts = torch.cat([torch.full((c,), c, device=device) for c in counts])
        r_weights = group_counts - orders
        r_update = sorted_d * r_weights.to(sorted_d.dtype)
        R_flat.scatter_add_(0, sorted_pixel_idx, r_update)
        N_flat.scatter_add_(0, pixel_idx, torch.ones_like(d))
    DA_flat = torch.zeros(H * W, device=device, dtype=torch.float32)
    nonzero = N_flat > 0
    DA_flat[nonzero] = R_flat[nonzero] / N_flat[nonzero]
    return DA_flat.view(H, W)
