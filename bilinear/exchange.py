
def tr2d(a):
    """ transpose last 2 dimensions of a """
    return a.transpose(range(a.ndim-2) + [-1,-2])


# edges
_N,_S,_E,_W = 0,1,2,3

# corners
_NE,_SE,_NW,_SW = 0,1,2,3

# opposing edge
#         N  S  E  W
_opp = [[_S,_N,_W,_E],  # unrotated connection
        [_W,_E,_S,_N]]  # rotated connection

# edge to the right
#          N  S  E  W
_redge = [_E,_W,_S,_N]

cslink = [[ (2,1), (5,0), (1,0), (4,1)],
          [ (2,0), (5,1), (3,1), (0,0)],
          [ (4,1), (1,0), (3,0), (0,1)],
          [ (4,0), (1,1), (5,1), (2,0)],
          [ (0,1), (3,0), (5,0), (2,1)],
          [ (0,0), (3,1), (1,1), (4,0)]]

llclink = [[ ( 1,0),   None, ( 3,0), (12,1)],  #  0
           [ ( 2,0), ( 0,0), ( 4,0), (11,1)],  #  1
           [ ( 6,1), ( 1,0), ( 5,0), (10,1)],  #  2
           [ ( 4,0),   None, ( 9,1), ( 0,0)],  #  3
           [ ( 5,0), ( 3,0), ( 8,1), ( 1,0)],  #  4
           [ ( 6,0), ( 4,0), ( 7,1), ( 2,0)],  #  5
           [ (10,1), ( 5,0), ( 7,0), ( 2,1)],  #  6
           [ (10,0), ( 5,1), ( 8,0), ( 6,0)],  #  7
           [ (11,0), ( 4,1), ( 9,0), ( 7,0)],  #  8
           [ (12,0), ( 3,1),   None, ( 8,0)],  #  9
           [ ( 2,1), ( 7,0), (11,0), ( 6,1)],  # 10
           [ ( 1,1), ( 8,0), (12,0), (10,0)],  # 11
           [ ( 0,1), ( 9,0),   None, (11,0)],  # 12
          ]

def mkcslink(faces=6, squeeze=True):
    """
    Return a list of cubed-sphere links (to be passed to Exchange).
    If faces is given, disconnect all faces not in this list
    (0-based; may be an integer, the number of faces starting from 0).
    If squeeze is True, exchange object will only contain connected faces.
    """
    try:
        iter(faces)
    except TypeError:
        faces = range(faces)
    # disconnect faces > nface
    if squeeze:
        link = [[ n in faces and (faces.index(n),e) or None for n,e in cslink[f] ]
                  for f in faces ]
    else:
        link = [[ f in faces and n in faces and (n,e) or None for n,e in links ]
                  for f,links in enumerate(cslink) ]
    return link


class Edge(object):
    def __init__(self,face,d):
        self.face = face
        self.d = d

    def right(self):
        dright = _redge[self.d]
        return self.face.edge[dright]

    def __str__(self):
        return "Edge(" + str(self.face.f) + "," + str(self.d) + ")"

    def __repr__(self):
        return "Edge(" + str(self.face.f) + "," + str(self.d) + ")"


class Face(object):
    def __init__(self, f, link):
        self.f = f
        self.nn = [ l and l[0] for l in link ]
        self.rot = [ l and l[1] or 0 for l in link ]
        self.d = [ _opp[rot][d] for d,rot in enumerate(self.rot) ]
        self.edge = [ Edge(self,d) for d in range(4) ]
        self.cycles = 4*[0]

class Exchange(object):
    def __init__(self, links):
        """
        Construct a new Exchange object given a list of edge links,

            links = [ face-0-links, face-1-links, ... ]
            face-n-links = [ N-link, S-link, E-link, W-link ]
            *-link = (opposing-face, rotate)

        where rotate is 1 if the opposing face needs to be rotated, 0 if not.
        See cslink for an example for cubed spheres.
        """
        self.links = links
        self.nf = len(links)
        self.faces = [ Face(f,l) for f,l in enumerate(links) ]
        for face in self.faces:
            for d in range(4):
                if face.nn[d] is not None:
                    face.edge[d].opp = self.faces[face.nn[d]].edge[face.d[d]]
                else:
                    face.edge[d].opp = None

        # lists of faces with "extra" vorticity points
        self.vcornersNW = []
        self.vcornersSE = []
        # indices of "extra" vorticity points (lookup table for face index)
        self.iNW = {}
        self.iSE = {}
        # contruct list of faces
        for face in self.faces:
            for d1,d2 in [(_E,_N),(_N,_W),(_W,_S),(_S,_E)]:
                faces = [face.f]
                e = face.edge[d1].opp
                n = 1
                while e and e.face != face:
                    faces.append(e.face.f)
                    e = e.right().opp
                    n += 1

                # number of faces connected to this corner (or 0 if not cyclic)
                face.cycles[d1] = e and n or 0

                # if cubed-sphere-like, check whether it is an "extra vorticity
                # point" not included in global mds files
                if 0 < face.cycles[d1] < 4:
                    if (d1,d2) == (_N,_W) and face.rot[_N]:
                        if faces[0] not in self.iNW:
                            for f in faces:
                                self.iNW[f] = len(self.vcornersNW)
                            self.vcornersNW.append(faces)
                    if (d1,d2) == (_S,_E) and face.rot[_S]:
                        if faces[0] not in self.iSE:
                            for f in faces:
                                self.iSE[f] = len(self.vcornersSE)
                            self.vcornersSE.append(faces)

    @classmethod
    def cs(cls, faces=6, squeeze=True):
        """
        Return a new cubed-sphere exchange object.
        If faces is given, disconnect all faces not in this list
        (0-based; may be an integer, the number of faces starting from 0).
        If squeeze is True, exchange object will only contain connected faces.
        """
        if faces == 6:
            link = cslink
        else:
            link = mkcslink(faces, squeeze)
        return cls(link)

    def find_color(self,f,dir):
        """not working"""
        faces = []
        dirs = []
        while f not in faces:
            faces.append(f)
            dirs.append(dir)
            f = self.face[f]

    def init_colors(self):
        """not working"""
        self.udir = [[] for f in range(self.nf)]
        self.vdir = [[] for f in range(self.nf)]
        while 1:
            for f in range(self.nf):
                if True not in self.udir[f]:
                    self.find_color(f,0)
                if True not in self.vdir[f]:
                    self.find_color(f,1)

    def tr(self, a, halo=1, corner=None, outside=None):
        """
        fill halo regions of c-point field a

        halo    :: width of halo
        corner  :: fill 3-valent (cs) halo corners with this value
                   None means "fill by double exchange" as in MITgcm (default)
                   False means "do not change"
        outside :: value to put in disconnected halos
        """
        for ipass in [0,1]:
            for f in range(self.nf):
                ny,nx = a[f].shape[-2:]
                # default is to fill full halo including corners
                h0S = 0
                h0N = 0
                h1S = 0
                h1N = 0
                if corner is False:
                    # do not update 3-valent corners
                    face = self.faces[f]
                    if face.cycles[_SW] < 4:
                        h0S = halo
                    if face.cycles[_NW] < 4:
                        h0N = halo
                    if face.cycles[_SE] < 4:
                        h1S = halo
                    if face.cycles[_NE] < 4:
                        h1N = halo
                # N
                lk = self.links[f][_N]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        a[f, ..., ny-halo:, h0N:nx-h1N] = tr2d(a[nn, ..., h1N:nx-h0N, halo:2*halo][..., ::-1, :])
                    else:
                        a[f, ..., ny-halo:, h0N:nx-h1N] = a[nn, ..., halo:2*halo, h0N:nx-h1N]
                elif outside is not None:
                    a[f, ..., ny-halo:, h0N:nx-h1N] = outside

                # S
                lk = self.links[f][_S]
                if lk is not None:
                    nn,rot = lk
                    nny,nnx = a[nn].shape[-2:]
                    if rot:
                        # this will serve as a check that dimensions agree
                        a[f, ..., :halo, h0S:nx-h1S] = tr2d(a[nn, ..., h1S:nx-h0S, nnx-2*halo:nnx-halo][..., ::-1, :])
                    else:
                        a[f, ..., :halo, h0S:nx-h1S] = a[nn, ..., nny-2*halo:nny-halo, h0S:nx-h1S]
                elif outside is not None:
                    a[f, ..., :halo, h0S:nx-h1S] = outside

            for f in range(self.nf):
                ny,nx = a[f].shape[-2:]
                h0W = 0
                h1W = 0
                h0E = 0
                h1E = 0
                if corner is False:
                    face = self.faces[f]
                    if face.cycles[_SW] < 4:
                        h0W = halo
                    if face.cycles[_NW] < 4:
                        h1W = halo
                    if face.cycles[_SE] < 4:
                        h0E = halo
                    if face.cycles[_NE] < 4:
                        h1E = halo
                # E
                lk = self.links[f][_E]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        a[f, ..., h0E:ny-h1E, nx-halo:] = tr2d(a[nn, ..., halo:2*halo, h1E:ny-h0E][..., ::-1])
                    else:
                        a[f, ..., h0E:ny-h1E, nx-halo:] = a[nn, ..., h0E:ny-h1E, halo:2*halo]
                elif outside is not None:
                    a[f, ..., h0E:ny-h1E, nx-halo:] = outside

                # W
                lk = self.links[f][_W]
                if lk is not None:
                    nn,rot = lk
                    nny,nnx = a[nn].shape[-2:]
                    if rot:
                        # this will serve as a check that dimensions agree
                        a[f, ..., h0W:ny-h1W, :halo] = tr2d(a[nn, ..., nny-2*halo:nny-halo, h1W:ny-h0W][..., ::-1])
                    else:
                        a[f, ..., h0W:ny-h1W, :halo] = a[nn, ..., h0W:ny-h1W, nnx-2*halo:nnx-halo]
                elif outside is not None:
                    a[f, ..., h0W:ny-n1W, :halo] = outside

        if corner is not None and corner is not False:
            for f,face in enumerate(self.faces):
                ny,nx = a[f].shape[-2:]
                if face.cycles[_SW] < 4: a[f, ...,    :halo,    :halo] = corner
                if face.cycles[_NW] < 4: a[f, ..., ny-halo:,    :halo] = corner
                if face.cycles[_SE] < 4: a[f, ...,    :halo, nx-halo:] = corner
                if face.cycles[_NE] < 4: a[f, ..., ny-halo:, nx-halo:] = corner

    __call__ = tr
    c = tr

    def ws(self, w, s, halo=0, extra=1, corner=None, outside=None, edge=None, corneredge=None):
        """
        fill halo regions of w-point field w and s-point field w without sign
        for rotated edges (good for masks, ...).  See uv for arguments.
        """
        self.uv(w, s, halo, extra, corner, outside, edge, corneredge, sign=1)

    def uv(self, u, v, halo=0, extra=1, corner=None, outside=None, edge=None, 
           corneredge=None, sign=-1):
        """
        fill halo regions of w-point field u and s-point field v

        halo    :: width of halo
        extra   :: extra halo for outgoing normal velocity (0 or 1)
        corner  :: fill in 3-valent halo corners and average adjacent velocities (???)
                   None means "fill by double exchange" as in MITgcm (default)
                   False is NOT YET IMPLEMENTED
        outside :: value to put in disconnected halos (exluding velocity on edge)
        edge    :: value to put in velocity on edge
        sign    :: sign to apply on rotated edges (-1 for velocities, +1 for masks, ...)
        """
        assert extra == u[0].shape[-1] - v[0].shape[-1]

        # normal velocities (tangential edges) first...
        for f in range(self.nf):
            ny,nx = v[f].shape[-2:]
            # N
            lk = self.links[f][_N]
            if lk is not None:
                nn,rot = lk
                if rot:
                    v[f, ..., ny-halo-extra:, :] = tr2d(u[nn, ..., ::-1, halo:2*halo+extra])
                else:
                    v[f, ..., ny-halo-extra:, :] = v[nn, ..., halo:2*halo+extra, :]
            else:
                if outside is not None:
                    v[f, ..., ny+1-halo-extra:, :] = outside
                if edge is not None:
                    v[f, ..., ny-halo-extra, :] = edge

            # S
            lk = self.links[f][_S]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = u[nn].shape[-2:]
                    v[f, ..., :halo, :] = tr2d(u[nn, ..., ::-1, ny-2*halo-extra:ny-halo-extra])
                else:
                    v[f, ..., :halo, :] = v[nn, ..., ny-2*halo-extra:ny-halo-extra, :]
            elif outside is not None:
                v[f, ..., :halo, :] = outside

        for f in range(self.nf):
            ny,nx = u[f].shape[-2:]
            # E
            lk = self.links[f][_E]
            if lk is not None:
                nn,rot = lk
                if rot:
                    u[f, ..., nx-halo-extra:] = tr2d(v[nn, ..., halo:2*halo+extra, ::-1])
                else:
                    u[f, ..., nx-halo-extra:] = u[nn, ..., halo:2*halo+extra]
            else:
                if outside is not None:
                    u[f, ..., nx+1-halo-extra:] = outside
                if edge is not None:
                    u[f, ..., nx-halo-extra] = edge

            # W
            lk = self.links[f][_W]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = v[nn].shape[-2:]
                    u[f, ..., :halo] = tr2d(v[nn, ..., nx-2*halo-extra:nx-halo-extra, ::-1])
                else:
                    u[f, ..., :halo] = u[nn, ..., nx-2*halo-extra:nx-halo-extra]
            elif outside is not None:
                u[f, ..., :halo] = outside

        # ...then tangential ones (normal edges)
        for f in range(self.nf):
            ny,nx = u[f].shape[-2:]
            # N
            lk = self.links[f][_N]
            if lk is not None:
                nn,rot = lk
                if rot:
                    u[f, ..., ny-halo:, 1-extra:] = sign*tr2d(v[nn, ..., :-extra and None:-1, halo:2*halo])
                else:
                    u[f, ..., ny-halo:, :] = u[nn, ..., halo:2*halo, :]
            elif outside is not None:
                u[f, ..., ny-halo:, :] = outside

            # S
            lk = self.links[f][_S]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = v[nn].shape[-2:]
                    u[f, ..., :halo, 1-extra:] = sign*tr2d(v[nn, ..., :-extra and None:-1, ny-2*halo:ny-halo])
                else:
                    u[f, ..., :halo, :] = u[nn, ..., ny-2*halo:ny-halo, :]
            elif outside is not None:
                u[f, ..., :halo, :] = outside

        for f in range(self.nf):
            ny,nx = v[f].shape[-2:]
            # E
            lk = self.links[f][_E]
            if lk is not None:
                nn,rot = lk
                if rot:
                    v[f, ..., 1-extra:, nx-halo:] = sign*tr2d(u[nn, ..., halo:2*halo, :-extra and None:-1])
                else:
                    v[f, ..., :, nx-halo:] = v[nn, ..., :, halo:2*halo]
            elif outside is not None:
                v[f, ..., :, nx-halo:] = outside

            # W
            lk = self.links[f][_W]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = u[nn].shape[-2:]
                    v[f, ..., 1-extra:, :halo] = sign*tr2d(u[nn, ..., nx-2*halo:nx-halo, :-extra and None:-1])
                else:
                    v[f, ..., :, :halo] = v[nn, ..., :, nx-2*halo:nx-halo]
            elif outside is not None:
                v[f, ..., :halo] = outside

        if corner is not None:
            for f,face in enumerate(self.faces):
                nyu,nxu = u[f].shape[-2:]
                nyv,nxv = v[f].shape[-2:]
                if face.cycles[_SW] < 4:
                    u[f, ..., :halo, :halo] = corner
                    v[f, ..., :halo, :halo] = corner
                if face.cycles[_NW] < 4:
                    u[f, ..., nyu-halo:,         :halo] = corner
                    v[f, ..., nyv-halo-extra+1:, :halo] = corner
                    v[f, ..., nyv-halo-extra, :halo] = .5*(v[f, ..., nyv-halo-extra, :halo] +
                                                           u[f, ..., nyu-2*halo:nyu-halo, halo])
                if face.cycles[_SE] < 4:
                    u[f, ...,    :halo, nxu+1-halo-extra:] = corner
                    v[f, ...,            :halo, nxv-halo:] = corner
                    ny,nx = u[f].shape[-2:]
                    u[f, ..., :halo, nxu-halo-extra] = .5*(u[f, ..., :halo, nxu-halo-extra] +
                                                           v[f, ..., halo, nxv-2*halo:nxv-halo])
                if face.cycles[_NE] < 4:
                    u[f, ..., nyu-halo:, nxu+1-halo-extra:] = corner
                    v[f, ..., nyv+1-halo-extra:, nxv-halo:] = corner
                    u[f, ..., nyu-halo:, nxu-halo-extra] = .5*(u[f, ..., nyu-halo:, nxu-halo-extra] +
                                                               v[f, ..., nyv-halo-extra, nxv-2*halo:nxv-halo])
                    v[f, ..., nyv-halo-extra, nxv-halo:] = .5*(v[f, ..., nyv-halo-extra, nxv-halo:] +
                                                               u[f, ..., nyu-2*halo:nyu-halo, nxu-halo-extra])

        if corneredge is not None:
            for f,face in enumerate(self.faces):
                nyu,nxu = u[f].shape[-2:]
                nyv,nxv = v[f].shape[-2:]
                if face.cycles[_SW] < 4:
                    u[f, ..., :halo, halo] = corneredge
                    v[f, ..., halo, :halo] = corneredge
                if face.cycles[_NW] < 4:
                    u[f, ..., nyu-halo:,       halo] = corneredge
                    v[f, ..., nyv-halo-extra, :halo] = corneredge
                if face.cycles[_SE] < 4:
                    u[f, ...,    :halo, nxu-halo-extra] = corneredge
                    v[f, ...,          halo, nxv-halo:] = corneredge
                if face.cycles[_NE] < 4:
                    u[f, ..., nyu-halo:, nxu-halo-extra] = corneredge
                    v[f, ..., nyv-halo-extra, nxv-halo:] = corneredge

    def uvB(self, u, v, halo=0, extra=1, corner=None, outside=None, edge=None,
            corneredge=None, sign=-1):
        """
        fill halo regions of B-grid velocity fields u and v

        halo    :: width of halo
        extra   :: extra halo for upper edges (0 or 1)
        corner  :: fill in 3-valent halo corners and average adjacent velocities (???)
                   None means "fill by double exchange" as in MITgcm (default)
                   False is NOT YET IMPLEMENTED
        outside :: value to put in disconnected halos (exluding velocity on edge)
        edge    :: value to put in velocity on edge
        sign    :: sign to apply on rotated edges (-1 for velocities, +1 for masks, ...)
        """
        e0 = 1-extra
        ee = None if extra else 0
        # normal velocities (tangential edges) first...
        for f in range(self.nf):
            ny,nx = v[f].shape[-2:]
            # N
            lk = self.links[f][_N]
            if lk is not None:
                nn,rot = lk
                if rot:
                    v[f, ..., ny-halo-extra:, e0:] = tr2d(u[nn, ..., :ee:-1, halo:2*halo+extra])
                else:
                    v[f, ..., ny-halo-extra:, :] = v[nn, ..., halo:2*halo+extra, :]
            else:
                if outside is not None:
                    v[f, ..., ny+1-halo-extra:, :] = outside
                if edge is not None:
                    v[f, ..., ny-halo-extra, :] = edge

            # S
            lk = self.links[f][_S]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = u[nn].shape[-2:]
                    v[f, ..., :halo, e0:] = tr2d(u[nn, ..., :ee:-1, ny-2*halo-extra:ny-halo-extra])
                else:
                    v[f, ..., :halo, :] = v[nn, ..., ny-2*halo-extra:ny-halo-extra, :]
            elif outside is not None:
                v[f, ..., :halo, :] = outside

        for f in range(self.nf):
            ny,nx = u[f].shape[-2:]
            # E
            lk = self.links[f][_E]
            if lk is not None:
                nn,rot = lk
                if rot:
                    u[f, ..., e0:, nx-halo-extra:] = tr2d(v[nn, ..., halo:2*halo+extra, :ee:-1])
                else:
                    u[f, ..., nx-halo-extra:] = u[nn, ..., halo:2*halo+extra]
            else:
                if outside is not None:
                    u[f, ..., nx+1-halo-extra:] = outside
                if edge is not None:
                    u[f, ..., nx-halo-extra] = edge

            # W
            lk = self.links[f][_W]
            if lk is not None:
                nn,rot = lk
                if rot:
                    nx,ny = v[nn].shape[-2:]
                    u[f, ..., e0:, :halo] = tr2d(v[nn, ..., nx-2*halo-extra:nx-halo-extra, :ee:-1])
                else:
                    u[f, ..., :halo] = u[nn, ..., nx-2*halo-extra:nx-halo-extra]
            elif outside is not None:
                u[f, ..., :halo] = outside

        # ...then tangential ones (normal edges)
        for f in range(self.nf):
            ny,nx = u[f].shape[-2:]
            # N
            lk = self.links[f][_N]
            if lk is not None:
                nn,rot = lk
                if rot:
                    u[f, ..., ny-halo-extra:, e0:] = sign*tr2d(v[nn, ..., :ee:-1, halo:2*halo+extra])
                else:
                    u[f, ..., ny-halo-extra:, :] = u[nn, ..., halo:2*halo+extra, :]
            else:
                if outside is not None:
                    u[f, ..., ny+1-halo-extra:, :] = outside
                if edge is not None:
                    u[f, ..., ny-halo-extra, :] = edge

            # S
            lk = self.links[f][_S]
            if lk is not None:
                nn,rot = lk
                if rot:
                    # this will serve as a check that dimensions agree
                    nx,ny = v[nn].shape[-2:]
                    u[f, ..., :halo, e0:] = sign*tr2d(v[nn, ..., :ee:-1, ny-2*halo:ny-halo])
                else:
                    u[f, ..., :halo, :] = u[nn, ..., ny-2*halo:ny-halo, :]
            elif outside is not None:
                u[f, ..., :halo, :] = outside

        for f in range(self.nf):
            ny,nx = v[f].shape[-2:]
            # E
            lk = self.links[f][_E]
            if lk is not None:
                nn,rot = lk
                if rot:
                    v[f, ..., e0:, nx-halo-extra:] = sign*tr2d(u[nn, ..., halo:2*halo+extra, :ee:-1])
                else:
                    v[f, ..., :, nx-halo-extra:] = v[nn, ..., :, halo:2*halo+extra]
            elif outside is not None:
                v[f, ..., :, nx+1-halo-extra:] = outside

            # W
            lk = self.links[f][_W]
            if lk is not None:
                nn,rot = lk
                if rot:
                    nx,ny = u[nn].shape[-2:]
                    v[f, ..., e0:, :halo] = sign*tr2d(u[nn, ..., nx-2*halo:nx-halo, :ee:-1])
                else:
                    v[f, ..., :, :halo] = v[nn, ..., :, nx-2*halo:nx-halo]
            elif outside is not None:
                v[f, ..., :halo] = outside

        if corner is not None:
            for f,face in enumerate(self.faces):
                ny,nx = u[f].shape[-2:]
                if face.cycles[_SW] < 4:
                    u[f, ..., :halo, :halo] = corner
                    v[f, ..., :halo, :halo] = corner
                if face.cycles[_NW] < 4:
                    u[f, ..., ny+1-halo-extra:, :halo] = corner
                    v[f, ..., ny-halo-extra+1:, :halo] = corner
                    v[f, ..., ny-halo-extra, :halo] = .5*(v[f, ..., ny-halo-extra, :halo] +
                                                          u[f, ..., ny-2*halo-extra:ny-halo-extra, halo])
                if face.cycles[_SE] < 4:
                    u[f, ..., :halo, nx-halo-extra+1:] = corner
                    v[f, ..., :halo, nx-halo-extra:] = corner
                    u[f, ..., :halo, nx-halo-extra] = .5*(u[f, ..., :halo, nx-halo-extra] +
                                                          v[f, ..., halo, nx-2*halo-extra:nx-halo-extra])
                if face.cycles[_NE] < 4:
                    u[f, ..., ny-halo-extra:, nx-halo-extra+1:] = corner
                    v[f, ..., ny-halo-extra+1:, nx-halo-extra:] = corner
                    u[f, ..., ny-halo-extra:, nx-halo-extra] = .5*(u[f, ..., ny-halo-extra:, nx-halo-extra] +
                                                                   v[f, ..., ny-halo-extra, nx-2*halo-extra:nx-halo-extra])
                    v[f, ..., ny-halo-extra, nx-halo-extra:] = .5*(v[f, ..., ny-halo-extra, nx-halo-extra:] +
                                                                   u[f, ..., ny-2*halo-extra:ny-halo-extra, nx-halo-extra])

        if corneredge is not None:
            for f,face in enumerate(self.faces):
                ny,nx = u[f].shape[-2:]
                if face.cycles[_SW] < 4:
                    u[f, ..., :halo, halo] = corneredge
                    v[f, ..., halo, :halo] = corneredge
                if face.cycles[_NW] < 4:
                    u[f, ..., ny-halo-extra:, halo] = corneredge
                    v[f, ..., ny-halo-extra, :halo] = corneredge
                if face.cycles[_SE] < 4:
                    u[f, ...,    :halo, nx-halo-extra] = corneredge
                    v[f, ...,    halo, nx-halo-extra:] = corneredge
                if face.cycles[_NE] < 4:
                    u[f, ..., ny-halo-extra:, nx-halo-extra] = corneredge
                    v[f, ..., ny-halo-extra, nx-halo-extra:] = corneredge

    def vort(self, a, aextra=None, halo=0, extra=1, corner=None, outside=None, edge=None):
        """
        fill halo regions of vorticity-point field a

        aextra  :: 2-tuple of lists of values for "missing" NW and SE corners
                   (1 each for cubed sphere: aextra=([aNW], [aSE]))
        halo    :: width of halo
        extra   :: extra halo at upper edge (0 or 1)
        corner  :: fill in 3-valent (cs) halo corners
                   None means "fill by double exchange" as in MITgcm (default)
                   False is NOT YET IMPLEMENTED
        outside :: value to put in disconnected halos
        edge    :: value to put in velocity on edge
        """
        e0 = 1-extra
        ee = None if extra else 0
        for ipass in [0,1]:
            for f in range(self.nf):
                ny,nx = a[f].shape[-2:]
                # N
                lk = self.links[f][_N]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        a[f, ..., ny-halo-extra:, e0:] = tr2d(a[nn, ..., :ee:-1, halo:2*halo+extra])
                    else:
                        a[f, ..., ny-halo-extra:, :] = a[nn, ..., halo:2*halo+extra, :]
                else:
                    if outside is not None:
                        a[f, ..., ny+1-halo-extra:, :] = outside
                    if edge is not None:
                        a[f, ..., ny-halo-extra, :] = edge

                # S
                lk = self.links[f][_S]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        # this will serve as a check that dimensions agree
                        nx,ny = a[nn].shape[-2:]
                        a[f, ..., :halo, e0:] = tr2d(a[nn, ..., :ee:-1, ny-2*halo-extra:ny-halo-extra])
                    else:
                        a[f, ..., :halo, :] = a[nn, ..., ny-2*halo-extra:ny-halo-extra, :]
                elif outside is not None:
                    a[f, ..., :halo, :] = outside

            for f in range(self.nf):
                ny,nx = a[f].shape[-2:]
                # E
                lk = self.links[f][_E]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        a[f, ..., e0:, nx-halo-extra:] = tr2d(a[nn, ..., halo:2*halo+extra, :ee:-1])
                    else:
                        a[f, ..., nx-halo-extra:] = a[nn, ..., halo:2*halo+extra]
                else:
                    if outside is not None:
                        a[f, ..., nx+1-halo-extra:] = outside
                    if edge is not None:
                        a[f, ..., nx-halo-extra] = edge

                # W
                lk = self.links[f][_W]
                if lk is not None:
                    nn,rot = lk
                    if rot:
                        nx,ny = a[nn].shape[-2:]
                        a[f, ..., e0:, :halo] = tr2d(a[nn, ..., nx-2*halo-extra:nx-halo-extra, :ee:-1])
                    else:
                        a[f, ..., :halo] = a[nn, ..., nx-2*halo-extra:nx-halo-extra]
                elif outside is not None:
                    a[f, ..., :halo] = outside

        if corner is not None:
            for f,face in enumerate(self.faces):
                ny,nx = a[f].shape[-2:]
                if face.cycles[_SW] < 4: a[f, ...,            :halo,            :halo] = corner
                if face.cycles[_NW] < 4: a[f, ..., ny+1-halo-extra:,            :halo] = corner
                if face.cycles[_SE] < 4: a[f, ...,            :halo, nx+1-halo-extra:] = corner
                if face.cycles[_NE] < 4: a[f, ..., ny+1-halo-extra:, nx+1-halo-extra:] = corner

        if aextra is not None:
            # fill in missing vorticity points
            aNW,aSE = aextra
            for f,i in self.iNW.items():
                a[f, ..., -halo-extra, halo] = aNW[i]

            for f,i in self.iSE.items():
                a[f, ..., halo, -halo-extra] = aSE[i]

    g = vort

    def coords(self, x, y, f, ncs, x0=0):
        for ipass in range(2):
            for ff in range(self.nf):
                # east
                msk = (f==ff) & (x > ncs + x0)
                fn,rot = self.links[ff][_E]
                if rot:
                    x[msk], y[msk] = ncs + 2*x0 - y[msk], x[msk] - ncs
                else:
                    x[msk] -= ncs
                f[msk] = fn

                # west
                msk = (f==ff) & (x < x0)
                fn,rot = self.links[ff][_W]
                if rot:
                    x[msk], y[msk] = ncs + 2*x0 - y[msk], x[msk] + ncs
                else:
                    x[msk] += ncs
                f[msk] = fn

                # north
                msk = (f==ff) & (y > ncs + x0)
                fn,rot = self.links[ff][_N]
                if rot:
                    x[msk], y[msk] = y[msk] - ncs, ncs + 2*x0 - x[msk]
                else:
                    y[msk] -= ncs
                f[msk] = fn

                # south
                msk = (f==ff) & (y < x0)
                fn,rot = self.links[ff][_S]
                if rot:
                    x[msk], y[msk] = y[msk] + ncs, ncs + 2*x0 - x[msk]
                else:
                    y[msk] += ncs
                f[msk] = fn


cs = Exchange.cs

def llc():
    return Exchange(llclink)


def printcs(a, w=3, d=0):
    ny,nx = a.shape[-2:]
    for j in range(a.nfacet//2-1, -1, -1):
        print
        pre = j*(nx*(w+1)+2)*' '
        for jj in range(ny-1, -1, -1):
            print pre, ' '.join('{:{}.{}f}'.format(x, w, d) for x in a[2*j, jj]), \
                  ' ', ' '.join('{:{}.{}f}'.format(x, w, d) for x in a[2*j+1, jj])


if __name__ == '__main__':
    link = [[ None , (2,0), (1,0), None ],
            [ None , (2,1), None , (0,0)],
            [ (0,0), None , (1,1), None ]]

    exch = Exchange(link)
    print exch.vcornersNW, exch.iNW
    print exch.vcornersSE, exch.iSE

    import fa
    a = fa.zeros(dims=3*[3,3])
    exch.vort(a, ([2.], [3.]))
    assert fa.np.all(a==0)

    csexch = cs()
    print csexch.vcornersNW, csexch.iNW
    print csexch.vcornersSE, csexch.iSE

    b = fa.zeros(dims=6*[3,3])
    csexch.vort(b, ([2.], [3.]))
    assert all(b[0::2,-1,0]==2.)
    assert all(b[1::2,0,-1]==3.)

