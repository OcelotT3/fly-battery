"""MDN direction test, per prereg-mdn-2026-09-27.md (sha256 b985aadf...). Appends one JSON line per (arm, stim, seed)."""
import sys, os, json, math, time
import numpy as np, torch
sys.path.insert(0, '/home/user/1f916-ai/1fab0/embodied')
torch.set_num_threads(4)
import evo_core as E, vnc_rate as V

OUT = sys.argv[1]; SEEDS = [int(s) for s in sys.argv[2:]] or [100, 101, 102, 103, 104]
W = '/home/user/1f916-ai/1fab0/embodied/results'
x_real = np.array(json.load(open(f'{W}/evo_real_v3/best.json'))['x']); x_shuf = np.array(json.load(open(f'{W}/evo_shuffled_v3/best.json'))['x'])
NP = len(x_real); RAND = np.random.default_rng(7).uniform(-2, 2, (5, NP))
STIM = {'DNg100': np.flatnonzero(V.TYP == 'DNg100'), 'MDN': np.flatnonzero(V.TYP == 'MDN')}
LEGNAME = ['L1', 'R2', 'L3', 'R1', 'L2', 'R3']          # E.LEGS order: flL, mlR, hlL, flR, mlL, hlR

def attach_pools(net):
    keep = net.keep; sup = net.sup
    def pool(leg, side, word):
        return np.flatnonzero((sup == 'vnc_motor') & (V.SUB[keep] == leg) & (V.SIDE[keep] == side) & np.array([word in t for t in V.TYP[keep]]))
    pools = {'LIFT': net.FLEX, 'DEP': net.EXT,
             'RET': [pool(l, s, 'Pleural remotor/abductor') for l, s in E.LEGS],
             'PRO': [pool(l, s, 'Tergopleural/Pleural promotor') for l, s in E.LEGS]}
    flat, offs = [], {}
    for k, lst in pools.items():
        for j, ix in enumerate(lst):
            offs[(k, j)] = (sum(len(a) for a in flat), len(ix)); flat.append(ix)
    net.rec_ix = torch.tensor(np.concatenate(flat), device=E.dev); net.offs = offs
    return {k: [len(a) for a in v] for k, v in pools.items()}

def analyse(net, Y, b, sat):
    seg = Y[400:, :, b]                                  # 0.4 s onward, 1 ms samples
    def m(k, j):
        o, n = net.offs[(k, j)]; return seg[:, o:o + n].mean(1) if n else None
    L = len(seg); fr = np.fft.rfftfreq(L, .001); band = (fr >= 5) & (fr <= 15)
    sig = np.array([m('LIFT', j) - m('DEP', j) for j in range(6)]); sig = sig - sig.mean(1, keepdims=True)
    P_ = np.abs(np.fft.rfft(sig, axis=1)) ** 2; k = np.flatnonzero(band)[P_[:, band].sum(0).argmax()]
    rhythm = float(P_[:, band].sum() / (P_[:, 1:].sum() + 1e-9))
    ph = np.angle(np.fft.rfft(sig, axis=1)[:, k]); grp = [0, 0, 0, 1, 1, 1]
    tripod = float(np.mean([math.cos(ph[p] - ph[q] - (0 if grp[p] == grp[q] else math.pi)) for p in range(6) for q in range(p + 1, 6)]))
    legs = {}
    for j in range(6):
        lift, ret, pro = m('LIFT', j), m('RET', j), m('PRO', j)
        c = lambda s: np.fft.rfft(s - s.mean())[k]
        pp = lambda s: float(s.max() - s.min())
        rec = dict(lift_pp=pp(lift), ret_pp=pp(ret) if ret is not None else None)
        if ret is None or pp(lift) < 5 or pp(ret) < 5: rec['D'] = None; rec['call'] = 'SILENT'
        else:
            D = math.cos(np.angle(c(lift)) - np.angle(c(ret))); rec['D'] = round(D, 3)
            rec['call'] = 'FORWARD' if D < -0.3 else 'BACKWARD' if D > 0.3 else 'UNCLEAR'
        if pro is not None and pp(pro) >= 5 and pp(lift) >= 5: rec['D_pro'] = round(math.cos(np.angle(c(lift)) - np.angle(c(pro))), 3)
        legs[LEGNAME[j]] = rec
    calls = [v['call'] for v in legs.values()]
    run = 'forward' if calls.count('FORWARD') >= 4 else 'backward' if calls.count('BACKWARD') >= 4 else 'neither'
    return dict(freq=float(fr[k]), rhythm=round(rhythm, 3), tripod=round(tripod, 3), sat=float(sat[b]), run=run, legs=legs)

nets = {'real': E.Net('real'), 'shuffled': E.Net('shuffled')}
sizes = {n: attach_pools(net) for n, net in nets.items()}
print('pool sizes (E.LEGS order)', sizes['real'], flush=True)
for sd in SEEDS:
    ri = sd - 100
    jobs = {'real': [('evolved_real', x_real), ('published_params', np.zeros(NP)), (f'random_real', RAND[ri % 5])],
            'shuffled': [('evolved_scrambled', x_shuf)]}
    for nn, arms in jobs.items():
        t0 = time.time(); cols = [(a, x, s) for a, x in arms for s in STIM]
        Y, sat = nets[nn].run(np.array([c[1] for c in cols]), sd, [STIM[c[2]] for c in cols], T=2.0)
        with open(OUT, 'a') as fh:
            for b, (a, x, s) in enumerate(cols):
                r = analyse(nets[nn], Y, b, sat); r.update(arm=a, stim=s, seed=sd); fh.write(json.dumps(r) + '\n')
                print(sd, a, s, r['run'], 'rhythm', r['rhythm'], 'tripod', r['tripod'], 'freq', r['freq'],
                      {k: v['D'] for k, v in r['legs'].items()}, flush=True)
        print(f'  ({nn}, {len(cols)} columns, {time.time() - t0:.0f}s)', flush=True)
