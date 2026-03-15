import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#Prinple: X=(X1, X2,...) -> Xt@X = X1t@X1 + X2t@X2 + ...
class PolyReg:

    def __init__(self, degree=2, ridge=1):
        self.degree = degree
        self.ridge = ridge
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

    def fit(self, X, y, nbatch=100):
        self.dim = X.shape[1]
        n = X.shape[0]
        batch_size = n // nbatch

        XtX = None
        Xty = None

        for i in range(nbatch):

            start = i * batch_size
            end = n if i == nbatch-1 else (i+1) * batch_size

            Xb = X[start:end]
            yb = y[start:end]

            Xp = self.poly.fit_transform(Xb)

            if XtX is None:
                p = Xp.shape[1]
                XtX = np.zeros((p, p))
                Xty = np.zeros(p)

            XtX += Xp.T @ Xp
            Xty += Xp.T @ yb
        XtX = XtX + self.ridge * np.eye(XtX.shape[0])
        self.coef_ = np.linalg.solve(XtX, Xty)

        return self

    def predict(self, X, nbatch=100):
        n = X.shape[0]
        batch_size = n // nbatch
        preds = []

        for i in range(nbatch):
            start = i * batch_size
            end = n if i == nbatch-1 else (i+1) * batch_size

            Xb = X[start:end]
            Xp = self.poly.transform(Xb)
            preds.append(Xp @ self.coef_)

        return np.concatenate(preds, axis=0)
    
    def get_interactions(self):
        d = self.dim + 1
        coef = self.coef_
        Q = np.zeros((d, d))

        k = 0

        for i in range(d):
            for j in range(i, d):
                Q[i, j] = coef[k]  
                Q[j, i] = coef[k]

                k += 1

        Q = np.log1p(np.abs(Q))
        return(Q)



def compare(dtype1, dtype2):
    """
    Return True if dtype1 is a floating type larger than dtype2.
    Only considers floating types.
    """
    if np.issubdtype(dtype1, float):
        return dtype1.itemsize > np.dtype(dtype2).itemsize
    return False
    
class To_Numpy:
    def __init__(self, ds, first_monday, n_weeks, float_tp):
        self.ds = ds
        self.fm = first_monday
        self.nw = n_weeks
        self.float_tp = float_tp
    def __call__(self, variable, dataset=None):
        if dataset == None:
            dataset = self.ds
        da = dataset[variable]
        if compare(da.dtype, self.float_tp):
            da = da.astype(self.float_tp, copy=False)
             # detect time dimension
        if "time" not in da.dims:
            return np.array(da.values)

        time_axis = da.dims.index("time")
        time_mode = "left" if time_axis == 0 else "right"

        fm, nw = self.fm, self.nw

        if time_mode == "right":
            return np.array(da)[:, fm:fm + nw*24*7]\
                .reshape(da.shape[0], nw, 7*24)

        if time_mode == "left":
            return np.array(da)[fm:fm + nw*24*7, :]\
                .reshape(nw, 7*24, da.shape[1])
        
def broadcast_to_shape(x, shape):
    """
    Broadcast tensor to a target shape (except last dimension).
    """

    if shape is None:
        return x

    target = []

    for xd, sd in zip(x.shape[:-1], shape):

        if xd == sd or xd == 1:
            target.append(sd)
        else:
            raise ValueError("Incompatible axis")

    target.append(x.shape[-1])

    return np.broadcast_to(x, target)


def concat_features(arrays, shape=None, flatten=False):
    """
    Concatenate features along the last axis with broadcasting.
    """

    if not flatten:
        return np.concatenate(
            [broadcast_to_shape(x, shape) for x in arrays],
            axis=-1
        )

    out = concat_features(arrays, shape=shape)

    dim1 = np.prod(out.shape[:-1])
    dim2 = out.shape[-1]

    return out.reshape(dim1, dim2)