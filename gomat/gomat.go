package gomat

import (
	"bytes"
	"io"
	"math"
	"math/rand"
	"os"
	"syscall"
	"unsafe"
)

type Matrix struct {
	data     []float32
	row, col int
}

func Zeros(row, col int) *Matrix {
	m := &Matrix{}
	m.data = make([]float32, row*col)
	m.row = row
	m.col = col
	return m
}

func (m *Matrix) Array() []float32 {
	return m.data
}

func (m *Matrix) GetRows(row_idxs []int) *Matrix {
	ret := &Matrix{}
	ret.col = m.col
	ret.row = len(row_idxs)
	ret.data = make([]float32, ret.col*ret.row)
	for k, v := range row_idxs {
		row := m.data[v*m.col : v*m.col+m.col]
		copy(ret.data[k*ret.col:], row)
	}
	return ret
}

func (m *Matrix) Transpose() *Matrix {
	ret := Zeros(m.col, m.row)
	for i := 0; i < m.Row(); i++ {
		for j := 0; j < m.Col(); j++ {
			ret.Set(j, i, m.Get(i, j))
		}
	}
	return ret
}

func (m *Matrix) Row() int {
	return m.row
}

func (m *Matrix) Col() int {
	return m.col
}

func (m *Matrix) Get(i, j int) float32 {
	return m.data[i*m.col+j]
}

func (m *Matrix) Set(i, j int, v float32) {
	m.data[i*m.col+j] = v
}

func addable(a, b *Matrix) {
	if a.row != b.row {
		panic("Row not equal!")
	}
	if b.col != b.col {
		panic("Col not equal!")
	}
}

func (m *Matrix) Add(b *Matrix) *Matrix {
	addable(m, b)
	c := Zeros(m.row, m.col)
	for k := range c.data {
		c.data[k] = m.data[k] + b.data[k]
	}
	return c
}

func (m *Matrix) Sub(b *Matrix) *Matrix {
	addable(m, b)
	c := Zeros(m.row, m.col)
	for k := range c.data {
		c.data[k] = m.data[k] - b.data[k]
	}
	return c
}

func (m *Matrix) SubEqual(b *Matrix) {
	addable(m, b)
	for k := range m.data {
		m.data[k] -= b.data[k]
	}
}

func (m *Matrix) MultFloat32(f float32) *Matrix {
	ret := Zeros(m.row, m.col)
	for k := range ret.data {
		ret.data[k] = m.data[k] * f
	}
	return ret
}

func (m *Matrix) Mult(B *Matrix) (C *Matrix) {
	k := m.col
	if k != B.row {
		panic("mult: Matrix doesnt match")
	}
	i := m.row
	j := B.col
	C = Zeros(i, j)
	matmult(nil, m, B, C, 0, i, 0, j, 0, k, 16)
	return C
}

//k是A的列数orB的行数
//i是A的行数
//j是B的列数
func matmult(done chan<- struct{}, A, B, C *Matrix, i0, i1, j0, j1, k0, k1, threshold int) {
	di := i1 - i0
	dj := j1 - j0
	dk := k1 - k0
	if di >= dj && di >= dk && di >= threshold {
		// divide in two by y axis
		mi := i0 + di/2
		done1 := make(chan struct{}, 1)
		go matmult(done1, A, B, C, i0, mi, j0, j1, k0, k1, threshold)
		matmult(nil, A, B, C, mi, i1, j0, j1, k0, k1, threshold)
		<-done1
	} else if dj >= dk && dj >= threshold {
		// divide in two by x axis
		mj := j0 + dj/2
		done1 := make(chan struct{}, 1)
		go matmult(done1, A, B, C, i0, i1, j0, mj, k0, k1, threshold)
		matmult(nil, A, B, C, i0, i1, mj, j1, k0, k1, threshold)
		<-done1
	} else if dk >= threshold {
		// divide in two by "k" axis
		// deliberately not parallel because of data races
		mk := k0 + dk/2
		matmult(nil, A, B, C, i0, i1, j0, j1, k0, mk, threshold)
		matmult(nil, A, B, C, i0, i1, j0, j1, mk, k1, threshold)
	} else {
		// the matrices are small enough, compute directly
		for i := i0; i < i1; i++ {
			for j := j0; j < j1; j++ {
				for k := k0; k < k1; k++ {
					C.data[i*C.col+j] += A.data[i*A.col+k] * B.data[k*B.col+j]
				}
			}
		}
	}
	if done != nil {
		done <- struct{}{}
	}
}

func (m *Matrix) Init_normal() {
	q := math.Sqrt(1.0 / float64(m.col))
	for k := range m.data {
		m.data[k] = float32(rand.NormFloat64() * q)
	}
}

func (m *Matrix) Map_npy_file(filename string) {
	header_sz := 80
	expected_sz := 4*m.Row()*m.Col() + header_sz
	fd, err := os.OpenFile(filename, os.O_RDWR, 0644)
	if err != nil {
		panic(err)
	}
	sz, err := fd.Seek(0, io.SeekEnd)
	if err != nil {
		panic(err)
	}
	if int(sz) != expected_sz {
		panic("data size mismatched!")
	}
	data, err := syscall.Mmap(int(fd.Fd()), 0, int(sz), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	if !bytes.Equal(data[1:6], []byte("NUMPY")) {
		panic("!NUMPY")
	}
	if data[header_sz-1] != '\n' {
		panic("!\\n")
	}
	for k := range m.data {
		m.data[k] = *(*float32)(unsafe.Pointer(&data[header_sz+4*k]))
	}
}

func RELU2(x, c *Matrix) *Matrix {
	addable(x, c)
	y := Zeros(x.row, c.col)
	for i := range y.data {
		if c.data[i] > 0 {
			y.data[i] = x.data[i]
		}
	}
	return y
}

func ARGMAX(a *Matrix) *Matrix {
	am := Zeros(a.row, 1)
	for i := 0; i < a.row; i++ {
		m := 0
		for j := 0; j < a.col; j++ {
			if a.data[i*a.col+m] < a.data[i*a.col+j] {
				m = j
			}
		}
		am.data[i] = float32(m)
	}
	return am
}

func LOGSOFTMAX(x *Matrix) *Matrix {
	lsm := Zeros(x.row, x.col)
	for i := 0; i < x.row; i++ {
		m := 0
		for j := 0; j < x.col; j++ {
			if x.Get(i, m) < x.Get(i, j) {
				m = j
			}
		}
		s := 0.0
		for j := 0; j < x.col; j++ {
			lsm.data[i*lsm.col+j] = x.data[i*x.col+j] - x.data[i*x.col+m]
			s += math.Exp(float64(lsm.Get(i, j)))
		}
		for j := 0; j < x.col; j++ {
			lsm.data[i*lsm.col+j] -= float32(math.Log(s))
		}
	}
	return lsm
}

func SOFTMAX(x *Matrix) *Matrix {
	y := LOGSOFTMAX(x)
	for k, v := range y.data {
		y.data[k] = float32(math.Exp(float64(v)))
	}
	return y
}

func SOFTMAX_CROSS_ENTROPY(x, c *Matrix) *Matrix {
	if c.col != 1 {
		panic("c.col != 1")
	}
	lsm := LOGSOFTMAX(x)
	smxe := Zeros(lsm.row, 1)
	for k := range smxe.data {
		smxe.data[k] = -lsm.data[k*lsm.col+int(c.data[k])]
	}
	return smxe
}
func SOFTMAX_MINUS_ONE(x, c *Matrix) *Matrix {
	if c.col != 1 {
		panic("c.col != 1")
	}
	y := SOFTMAX(x)
	for i := 0; i < y.row; i++ {
		y.data[i*y.col+int(c.data[i])] -= 1.0
	}
	return y
}
