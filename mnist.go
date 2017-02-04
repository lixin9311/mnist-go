package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/dterei/gotsc"

	"./gomat"
)

const (
	n_classes       = 10
	pixel_height    = 28
	pixel_width     = 28
	n_pixels        = pixel_height * pixel_width
	n_training_data = 60000
	n_test_data     = 10000

	n_units      = 100
	max_batch_sz = 1000
)

type mnist_opt struct {
	samples_to_train int
	/* total number of samples in the test phase */
	samples_to_test int
	/* learning rate */
	eta float64
	/* batch size */
	batch_sz int
	/* control how often it shows progress
	   (recent classificatin score and cross entropy error) */
	progress_interval int
	/* the number of last samples used for scoring
	   classificationn/cross-entropy error */
	average_window_sz int
	/* seed of the random number generator used to initialize weight matrices
	   (initial weight matrices are deterministic given the same seed) */
	weight_seed int64
	/* seed of the random number generator used to draw random samples in the training
	   (drawn samples are deterministic given the same seed) */
	draw_seed uint
	/* prefix of filenames to write samples to */
	sample_image_prefix string
	/* max number of sample image files */
	max_sample_images int
	/* the factor by which images are magnified (1 for no magnification) */
	magnify_images int
}

func NewMnistOpt() *mnist_opt {
	ret := &mnist_opt{}
	ret.samples_to_train = 60000
	ret.samples_to_test = 10000
	ret.eta = 0.005
	ret.batch_sz = 100
	ret.weight_seed = 729918723710
	ret.draw_seed = 314159265358
	ret.progress_interval = 1000
	ret.average_window_sz = 10000
	ret.sample_image_prefix = "imgs/sample"
	ret.max_sample_images = 0
	ret.magnify_images = 1
	return ret
}

func train_mnist(W0, W1, W2 *gomat.Matrix, opt *mnist_opt) float64 {
	/* files to read training data (and their classes) from */
	train_x := "data/train_images_60000x784_float32.npy" /* training data */
	train_y := "data/train_labels_60000_float32.npy"     /* their classes */
	X := gomat.Zeros(n_training_data, n_pixels)
	C := gomat.Zeros(n_training_data, 1)
	X.Map_npy_file(train_x)
	C.Map_npy_file(train_y)

	n_samples_to_train := opt.samples_to_train
	eta := float32(opt.eta)
	sc := NewSC(opt.average_window_sz)
	progress_interval := opt.progress_interval
	bs := opt.batch_sz

	t0 := time.Now()
	c0 := gotsc.BenchStart()
	for i := 0; i < n_samples_to_train; i += bs {
		var ns int
		if i+bs <= n_samples_to_train {
			ns = bs
		} else {
			ns = n_samples_to_train - i
		}
		samples := make([]int, ns)
		choose_random_samples(n_training_data, samples)
		x := X.GetRows(samples)
		c := C.GetRows(samples)

		for d := 0; d < ns; d++ {
			if i+d >= opt.max_sample_images {
				break
			}
			filename := fmt.Sprintf("%s%06d_%d.ppm", opt.sample_image_prefix, i+d, int(c.Get(d, 0)))
			dump_as_ppm(x.Get(d, 0), pixel_height, pixel_width, opt.magnify_images, filename)
		}

		x1 := x.Mult(W0)
		x2 := gomat.RELU2(x1, x1)

		x3 := x2.Mult(W1)
		x4 := gomat.RELU2(x3, x3)

		x5 := x4.Mult(W2)
		p := gomat.ARGMAX(x5)
		e := gomat.SOFTMAX_CROSS_ENTROPY(x5, c)

		g_x5 := gomat.SOFTMAX_MINUS_ONE(x5, c)
		g_x4 := g_x5.Mult(W2.Transpose())
		g_x3 := gomat.RELU2(g_x4, x3)
		g_x2 := g_x3.Mult(W1.Transpose())
		g_x1 := gomat.RELU2(g_x2, x1)
		g_w2 := x4.Transpose().Mult(g_x5)
		g_w1 := x2.Transpose().Mult(g_x3)
		g_w0 := x.Transpose().Mult(g_x1)

		W0.SubEqual(g_w0.MultFloat32(eta))
		W1.SubEqual(g_w1.MultFloat32(eta))
		W2.SubEqual(g_w2.MultFloat32(eta))

		sc.Update_score(p, c, e)

		i_next := i + ns
		if opt.progress_interval > 0 && i_next/progress_interval > i/progress_interval {
			c1 := gotsc.BenchEnd()
			dc := c1 - c0
			dt := time.Since(t0)
			var tr int
			if i_next-sc.n > 0 {
				tr = i_next - sc.n
			}
			fmt.Printf("training %d - %d at %d clocks and %v : classification %d / %d = %f cross entropy error = %.8f\n",
				tr, i_next, dc, dt, sc.sum_c, sc.n, float64(sc.sum_c)/float64(sc.n), sc.sum_e/float64(sc.n))
		}
	}
	return 2.0 * (2.0*float64(n_pixels*n_units) + 3.0*float64(n_units*n_units) + 3.0*float64(n_units*n_classes)) * float64(n_samples_to_train)
}

func test_mnist(W0, W1, W2 *gomat.Matrix, opt *mnist_opt) float64 {
	test_x := "data/test_images_10000x784_float32.npy" /* training data */
	test_y := "data/test_labels_10000_float32.npy"     /* their classes */
	X := gomat.Zeros(n_test_data, n_pixels)
	C := gomat.Zeros(n_test_data, 1)
	X.Map_npy_file(test_x)
	C.Map_npy_file(test_y)

	n_samples_to_test := opt.samples_to_test
	sc := NewSC(opt.average_window_sz)
	bs := opt.batch_sz

	t0 := time.Now()
	c0 := gotsc.BenchStart()
	for i := 0; i < n_samples_to_test; i += bs {
		var ns int
		if i+bs <= n_samples_to_test {
			ns = bs
		} else {
			ns = n_samples_to_test - i
		}
		samples := make([]int, ns)
		get_seq_samples(n_test_data, i, samples)
		x := X.GetRows(samples)
		c := C.GetRows(samples)

		x1 := x.Mult(W0)
		x2 := gomat.RELU2(x1, x1)

		x3 := x2.Mult(W1)
		x4 := gomat.RELU2(x3, x3)

		x5 := x4.Mult(W2)

		e := gomat.SOFTMAX_CROSS_ENTROPY(x5, c)
		y := gomat.ARGMAX(x5)
		sc.Update_score(y, c, e)
	}
	dc := gotsc.BenchEnd() - c0
	dt := time.Since(t0)
	var test int
	if n_samples_to_test-sc.n > 0 {
		test = n_samples_to_test - sc.n
	}
	fmt.Printf("test %d - %d at %d clocks and %v : classification %d / %d = %f cross entropy error = %.8f\n",
		test, n_samples_to_test, dc, dt, sc.sum_c, sc.n, float64(sc.sum_c)/float64(sc.n), sc.sum_e/float64(sc.n))
	return 2.0 * (1.0*float64(n_pixels*n_units) + 1.0*float64(n_units*n_units) + 1.0*float64(n_units*n_classes)) * float64(n_samples_to_test)
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	opt := NewMnistOpt()
	rand.Seed(opt.weight_seed)
	W0 := gomat.Zeros(n_pixels, n_units)
	W1 := gomat.Zeros(n_units, n_units)
	W2 := gomat.Zeros(n_units, n_classes)
	W0.Init_normal()
	W1.Init_normal()
	W2.Init_normal()

	fmt.Printf("samples for training: %v\n", opt.samples_to_train)
	fmt.Printf("samples for testing: %v\n", opt.samples_to_test)
	fmt.Printf("mini batch size: %v\n", opt.batch_sz)
	fmt.Printf("learning rate: %f\n", opt.eta)
	fmt.Printf("max mini batch size: %d\n", max_batch_sz)
	fmt.Printf("hidden units: %d\n", n_units)
	fmt.Printf("pixels per image: %d\n", n_pixels)
	fmt.Printf("output classes: %d\n", n_classes)

	fmt.Printf("seed for weight: %v\n", opt.weight_seed)
	fmt.Printf("seed for drawing samples: %v\n", opt.draw_seed)
	fmt.Printf("interval between reports: %v\n", opt.progress_interval)
	fmt.Printf("average window size: %v\n", opt.average_window_sz)
	fmt.Printf("sample images written: %v\n", opt.max_sample_images)
	if opt.max_sample_images > 0 {
		fmt.Printf("  with prefix: %s\n", opt.sample_image_prefix)
	}
	fmt.Printf("images are magnified by a factor: %v\n", opt.magnify_images)
	fmt.Printf("language: %s\n", "Golang")
	fmt.Printf("library: %s\n", "none")
	tsc := gotsc.TSCOverhead()

	t0 := time.Now()
	c0 := gotsc.BenchStart()
	flops_train := train_mnist(W0, W1, W2, opt)
	c1 := gotsc.BenchEnd()
	t1 := time.Now()

	dc_train := (c1 - c0 - tsc)
	dt_train := t1.Sub(t0)

	t2 := time.Now()
	c2 := gotsc.BenchStart()
	flops_test := test_mnist(W0, W1, W2, opt)
	c3 := gotsc.BenchEnd()
	t3 := time.Now()

	dc_test := (c3 - c2 - tsc)
	dt_test := t3.Sub(t2)
	fmt.Printf("%.0f flops in %d clocks / %v to train (%.2f flops/clock)\n",
		flops_train, dc_train, dt_train, flops_train/float64(dc_train))
	fmt.Printf("%.0f flops in %d clocks / %v to test (%.2f flops/clock)\n",
		flops_test, dc_test, dt_test, flops_test/float64(dc_test))

	// fmt.Println(c_test, c_train, t_test, t_train)
	// C := gomat.Zeros(n_training_data, 1)
	// train_y := "data/train_labels_60000_float32.npy"
	// C.Map_npy_file(train_y)
	// fmt.Println(C)
}
