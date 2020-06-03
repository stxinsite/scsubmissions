import numpy as np

def calcerr(ref,test):
    mag = 0.0
    err = 0.0
    for x in range(128):
        for y in range(128):
            for z in range(128):
                rv = ref[x][y][z]
                tv = test[x][y][z]
                ev = rv - tv
                mag += np.real(rv)*np.real(rv) + np.imag(rv)*np.imag(rv)
                err += np.real(ev)*np.real(ev) + np.imag(ev)*np.imag(ev)
    return (mag, err, 10.0 * np.log10(err/mag))

with open('derived/pmeinput.bin', 'r') as f:
    data=np.fromfile(f, np.single)
pmeinputsingle = np.reshape(data,(128,128,128))

pmeinput = np.zeros((128,128,128),np.csingle)
for x in range(128):
    for y in range(128):
        for z in range(128):
            pmeinput[x][y][z] = complex(pmeinputsingle[x][y][z], 0.0)

with open('derived/pmeoutput.bin', 'r') as f:
    data=np.fromfile(f, np.csingle)
pmeoutput = np.reshape(data,(128,128,128))

with open('derived/eterm.bin', 'r') as f:
    data=np.fromfile(f, np.single)
eterm = np.reshape(data,(128,128,128))

scaleconst = np.ones((128,128,128),np.single)

pmefft = np.fft.fftn(pmeinput)
scaled = np.zeros((128,128,128),np.csingle)

for x in range(128):
    for y in range(128):
        for z in range(128):
            scaled[x][y][z] = pmefft[x][y][z] * scaleconst[x][y][z]

result = np.fft.ifftn(scaled)
print("fb", calcerr(pmeinput, result))

for x in range(128):
    for y in range(128):
        for z in range(128):
            scaled[x][y][z] = pmefft[x][y][z] * eterm[x][y][z] * (128.0 * 128.0 * 128.0)

result = np.fft.ifftn(scaled)
print("pme", calcerr(pmeoutput, result))

#print "pmeoutput", pmeoutput[0:10][0][0]
#print "result", result[0:10][0][0]
#print "ratio", np.divide(pmeoutput[0:10][0][0], result[0:10][0][0])
