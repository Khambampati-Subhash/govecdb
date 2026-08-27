//go:build ignore

// Command plot renders results.csv into the SVG charts embedded in the READMEs.
//
// It is deliberately not part of the module build (`//go:build ignore`): GoVecDB
// is a library with no third-party dependencies and no commands, and a charting
// dependency would have been the first crack in that. Everything here is
// stdlib — the SVG is written by hand, which for axes, gridlines and polylines
// is a couple hundred lines and no supply chain at all.
//
// Regenerate both halves together, so the numbers and the pictures never drift:
//
//	go test ./internal/hnsw/ -run TestSweep -results docs/benchmarks/results.csv -timeout 30m
//	go run docs/benchmarks/plot.go
//
// The palette matches docs/diagrams: dark canvas, and the same hues carry the
// same meaning across every image in the repo.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------- palette

const (
	colBG      = "#000000"
	colPanel   = "#0C0C14"
	colGrid    = "#26263A"
	colAxis    = "#6E6E8A"
	colText    = "#CDD6F4"
	colMuted   = "#9494B8"
	colBlue    = "#4FC3F7" // RAM / the primary measurement, as in the diagrams
	colOrange  = "#FFB74D" // WAL / the cost being traded against
	colGreen   = "#66BB6A" // disk / the "after" state
	colRed     = "#FF6B6B" // critical
	colPurple  = "#B39DDB" // API surface
	colYellow  = "#FFD54F" // decision / caveat
	fontFamily = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
)

// ---------------------------------------------------------------- data

type sample struct {
	Sweep  string
	Label  string
	Dim    int
	N      int
	M      int
	Ef     int
	K      int
	Recall float64
	Search float64 // nanoseconds per query
	Build  float64 // milliseconds for the corpus
}

func (s sample) searchMicros() float64 { return s.Search / 1000 }

func readSamples(path string) []sample {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open %s: %v (run the -results sweep first)", path, err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	if len(rows) < 2 {
		log.Fatalf("%s has no data rows", path)
	}

	atoi := func(s string) int { n, _ := strconv.Atoi(s); return n }
	atof := func(s string) float64 { f, _ := strconv.ParseFloat(s, 64); return f }

	var out []sample
	for _, r := range rows[1:] { // skip header
		if len(r) < 10 {
			continue
		}
		out = append(out, sample{
			Sweep: r[0], Label: r[1],
			Dim: atoi(r[2]), N: atoi(r[3]), M: atoi(r[4]),
			Ef: atoi(r[5]), K: atoi(r[6]),
			Recall: atof(r[7]), Search: atof(r[8]), Build: atof(r[9]),
		})
	}
	return out
}

func bySweep(all []sample, name string) []sample {
	var out []sample
	for _, s := range all {
		if s.Sweep == name {
			out = append(out, s)
		}
	}
	return out
}

// sortNumericLabels orders cells by their label read as a number, so an axis
// runs 8, 32, 128 rather than "128", "32", "8".
func sortNumericLabels(ss []sample) {
	sort.SliceStable(ss, func(i, j int) bool {
		a, errA := strconv.Atoi(strings.TrimSuffix(ss[i].Label, "%"))
		b, errB := strconv.Atoi(strings.TrimSuffix(ss[j].Label, "%"))
		if errA != nil || errB != nil {
			return ss[i].Label < ss[j].Label
		}
		return a < b
	})
}

// ---------------------------------------------------------------- chart model

type series struct {
	name   string
	colour string
	values []float64
	right  bool // plot against the secondary axis
	bar    bool
}

type axis struct {
	label  string
	min    float64
	max    float64
	step   float64
	format func(float64) string
}

func (a axis) ticks() []float64 {
	var out []float64
	for v := a.min; v <= a.max+a.step/2; v += a.step {
		out = append(out, v)
	}
	return out
}

type chart struct {
	title      string
	subtitle   string
	xLabel     string
	categories []string
	left       axis
	right      *axis
	series     []series
	note       string
}

const (
	width      = 900
	height     = 460
	padTop     = 96
	padBottom  = 74
	padLeft    = 78
	padRight   = 86
	plotWidth  = width - padLeft - padRight
	plotHeight = height - padTop - padBottom
)

func (c *chart) xCentre(i int) float64 {
	step := float64(plotWidth) / float64(len(c.categories))
	return float64(padLeft) + step*(float64(i)+0.5)
}

func (a axis) y(v float64) float64 {
	if a.max == a.min {
		return float64(padTop + plotHeight)
	}
	frac := (v - a.min) / (a.max - a.min)
	return float64(padTop+plotHeight) - frac*float64(plotHeight)
}

// niceAxis picks a rounded min/max and a tick step that lands on readable
// numbers. It aims for six intervals rather than four: the 1/2/2.5/5 ladder is
// coarse, and asking it to cover a range in four steps rounds up hard — a recall
// span of 0.31..1.00 came out as an axis running to 2.000.
func niceAxis(label string, values []float64, zeroBased bool, format func(float64) string) axis {
	lo, hi := values[0], values[0]
	for _, v := range values {
		lo, hi = minF(lo, v), maxF(hi, v)
	}
	if zeroBased {
		lo = 0
	} else {
		lo -= (hi - lo) * 0.30 // margin, so the lowest point is not on the axis
		if lo < 0 {
			lo = 0
		}
	}
	hi += (hi - lo) * 0.10
	if hi == lo {
		hi = lo + 1
	}

	step := niceStep((hi - lo) / 6)
	lo = math.Floor(lo/step) * step
	hi = math.Ceil(hi/step) * step
	return axis{label: label, min: lo, max: hi, step: step, format: format}
}

// recallAxis is a fixed-domain version: recall cannot exceed 1, so the top of
// the axis is 1 regardless of the data and only the floor moves. That keeps the
// same quantity comparable by eye across charts, which a data-fitted axis would
// quietly destroy.
func recallAxis(label string, values []float64) axis {
	lo := values[0]
	for _, v := range values {
		lo = minF(lo, v)
	}
	lo = math.Floor((lo-0.05)*10) / 10
	if lo < 0 {
		lo = 0
	}
	step := niceStep((1 - lo) / 5)
	lo = math.Floor(lo/step) * step
	return axis{label: label, min: lo, max: 1, step: step, format: fmtRecall}
}

func niceStep(raw float64) float64 {
	mag := 1.0
	for raw >= 10 {
		raw /= 10
		mag *= 10
	}
	for raw < 1 {
		raw *= 10
		mag /= 10
	}
	switch {
	case raw <= 1:
		return 1 * mag
	case raw <= 2:
		return 2 * mag
	case raw <= 2.5:
		return 2.5 * mag
	case raw <= 5:
		return 5 * mag
	default:
		return 10 * mag
	}
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// ---------------------------------------------------------------- rendering

func (c *chart) render() string {
	var b strings.Builder

	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" aria-label="%s">`,
		width, height, width, height, esc(c.title))
	fmt.Fprintf(&b, `<rect width="%d" height="%d" fill="%s"/>`, width, height, colBG)
	fmt.Fprintf(&b, `<rect x="8" y="8" width="%d" height="%d" rx="10" fill="%s"/>`, width-16, height-16, colPanel)

	// Titles.
	text(&b, 26, 38, colText, 19, "start", "600", c.title)
	if c.subtitle != "" {
		text(&b, 26, 58, colMuted, 12.5, "start", "400", c.subtitle)
	}

	c.renderGrid(&b)
	c.renderBars(&b)
	c.renderLines(&b)
	c.renderXAxis(&b)
	c.renderLegend(&b)

	if c.note != "" {
		text(&b, 26, height-16, colMuted, 12, "start", "400", c.note)
	}
	b.WriteString(`</svg>`)
	return b.String()
}

func (c *chart) renderGrid(b *strings.Builder) {
	for _, v := range c.left.ticks() {
		y := c.left.y(v)
		fmt.Fprintf(b, `<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1"/>`,
			padLeft, y, padLeft+plotWidth, y, colGrid)
		text(b, float64(padLeft)-10, y+4, colMuted, 11.5, "end", "400", c.left.format(v))
	}
	// Axis titles: the left one is rotated up the side, as usual.
	fmt.Fprintf(b, `<text x="%.1f" y="%.1f" fill="%s" font-family="%s" font-size="12" text-anchor="middle" transform="rotate(-90 %.1f %.1f)">%s</text>`,
		20.0, float64(padTop+plotHeight/2), colText, fontFamily,
		20.0, float64(padTop+plotHeight/2), esc(c.left.label))

	if c.right != nil {
		for _, v := range c.right.ticks() {
			text(b, float64(padLeft+plotWidth)+10, c.right.y(v)+4, colMuted, 11.5, "start", "400", c.right.format(v))
		}
		fmt.Fprintf(b, `<text x="%.1f" y="%.1f" fill="%s" font-family="%s" font-size="12" text-anchor="middle" transform="rotate(90 %.1f %.1f)">%s</text>`,
			float64(width-18), float64(padTop+plotHeight/2), colText, fontFamily,
			float64(width-18), float64(padTop+plotHeight/2), esc(c.right.label))
	}

	// Baseline.
	fmt.Fprintf(b, `<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>`,
		padLeft, padTop+plotHeight, padLeft+plotWidth, padTop+plotHeight, colAxis)
}

func (c *chart) axisFor(s series) axis {
	if s.right && c.right != nil {
		return *c.right
	}
	return c.left
}

func (c *chart) renderBars(b *strings.Builder) {
	var bars []series
	for _, s := range c.series {
		if s.bar {
			bars = append(bars, s)
		}
	}
	if len(bars) == 0 {
		return
	}

	slot := float64(plotWidth) / float64(len(c.categories))
	groupW := slot * 0.62
	barW := groupW / float64(len(bars))

	for si, s := range bars {
		ax := c.axisFor(s)
		for i, v := range s.values {
			x := c.xCentre(i) - groupW/2 + barW*float64(si)
			y := ax.y(v)
			h := float64(padTop+plotHeight) - y
			if h < 0 {
				h = 0
			}
			fmt.Fprintf(b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" fill="%s" fill-opacity="0.85"/>`,
				x, y, barW*0.86, h, s.colour)
			text(b, x+barW*0.43, y-7, colText, 11, "middle", "600", ax.format(v))
		}
	}
}

func (c *chart) renderLines(b *strings.Builder) {
	for _, s := range c.series {
		if s.bar {
			continue
		}
		ax := c.axisFor(s)

		var pts []string
		for i, v := range s.values {
			pts = append(pts, fmt.Sprintf("%.1f,%.1f", c.xCentre(i), ax.y(v)))
		}
		fmt.Fprintf(b, `<polyline points="%s" fill="none" stroke="%s" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`,
			strings.Join(pts, " "), s.colour)

		for i, v := range s.values {
			x, y := c.xCentre(i), ax.y(v)
			fmt.Fprintf(b, `<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="%s" stroke-width="2"/>`,
				x, y, colPanel, s.colour)
			// Alternate the label side so two series stay readable, then flip
			// any label that would run into the legend above the plot or off
			// the baseline below it.
			dy := -12.0
			if s.right {
				dy = 20.0
			}
			if y+dy < float64(padTop)+2 {
				dy = 18.0
			} else if y+dy > float64(padTop+plotHeight)+4 {
				dy = -12.0
			}
			text(b, x, y+dy, colText, 11, "middle", "600", ax.format(v))
		}
	}
}

func (c *chart) renderXAxis(b *strings.Builder) {
	for i, cat := range c.categories {
		text(b, c.xCentre(i), float64(padTop+plotHeight)+22, colText, 12, "middle", "400", cat)
	}
	text(b, float64(padLeft+plotWidth/2), float64(padTop+plotHeight)+46, colMuted, 12, "middle", "400", c.xLabel)
}

func (c *chart) renderLegend(b *strings.Builder) {
	x := float64(padLeft)
	y := 78.0
	for _, s := range c.series {
		fmt.Fprintf(b, `<rect x="%.1f" y="%.1f" width="20" height="4" rx="2" fill="%s"/>`, x, y-4, s.colour)
		text(b, x+27, y, colText, 12, "start", "500", s.name)
		x += 27 + float64(len(s.name))*7 + 26
	}
}

func text(b *strings.Builder, x, y float64, fill string, size float64, anchor, weight, body string) {
	fmt.Fprintf(b, `<text x="%.1f" y="%.1f" fill="%s" font-family="%s" font-size="%.1f" text-anchor="%s" font-weight="%s">%s</text>`,
		x, y, fill, fontFamily, size, anchor, weight, esc(body))
}

func esc(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;").Replace(s)
}

func write(name, svg string) {
	path := filepath.Join("docs", "benchmarks", name)
	if err := os.WriteFile(path, []byte(svg), 0o644); err != nil {
		log.Fatalf("write %s: %v", path, err)
	}
	fmt.Println("wrote", path)
}

// ---------------------------------------------------------------- helpers

func labels(ss []sample) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = s.Label
	}
	return out
}

func recalls(ss []sample) []float64 {
	out := make([]float64, len(ss))
	for i, s := range ss {
		out[i] = s.Recall
	}
	return out
}

func micros(ss []sample) []float64 {
	out := make([]float64, len(ss))
	for i, s := range ss {
		out[i] = s.searchMicros()
	}
	return out
}

func buildSecs(ss []sample) []float64 {
	out := make([]float64, len(ss))
	for i, s := range ss {
		out[i] = s.Build / 1000
	}
	return out
}

func fmtRecall(v float64) string { return strconv.FormatFloat(v, 'f', 3, 64) }
func fmtMicros(v float64) string { return strconv.FormatFloat(v, 'f', 0, 64) + "µs" }
func fmtSecs(v float64) string   { return strconv.FormatFloat(v, 'f', 1, 64) + "s" }

func corpusNote(ss []sample) string {
	if len(ss) == 0 {
		return ""
	}
	return fmt.Sprintf("N=%d vectors · k=%d · recall measured against brute-force ground truth",
		ss[0].N, ss[0].K)
}

// ---------------------------------------------------------------- charts

func main() {
	all := readSamples(filepath.Join("docs", "benchmarks", "results.csv"))

	plotDimension(bySweep(all, "dimension"))
	plotScale(bySweep(all, "scale"))
	plotEf(bySweep(all, "ef"))
	plotM(bySweep(all, "M"))
	plotMetric(bySweep(all, "metric"))
	plotDistribution(bySweep(all, "distribution"))
	plotTombstones(bySweep(all, "tombstones"), bySweep(all, "compacted"))
}

func plotDimension(ss []sample) {
	if len(ss) == 0 {
		return
	}
	sortNumericLabels(ss)

	c := &chart{
		title:      "Recall and latency across dimensions",
		subtitle:   "Recall falls as dimension rises — distances concentrate, so a neighbourhood says less about where to go next.",
		xLabel:     "vector dimension",
		categories: labels(ss),
		left:       recallAxis("recall@10", recalls(ss)),
		series: []series{
			{name: "recall@10", colour: colBlue, values: recalls(ss)},
			{name: "search latency", colour: colOrange, values: micros(ss), right: true},
		},
		note: corpusNote(ss) + " · ef=64",
	}
	right := niceAxis("µs per query", micros(ss), true, fmtMicros)
	c.right = &right
	write("recall-vs-dimension.svg", c.render())
}

func plotScale(ss []sample) {
	if len(ss) == 0 {
		return
	}
	sortNumericLabels(ss)

	c := &chart{
		title:      "Search latency against corpus size",
		subtitle:   "A search visits a tiny fraction of the graph, so latency tracks log N. Build time is the part that grows with N.",
		xLabel:     "vectors in the index",
		categories: labels(ss),
		left:       niceAxis("µs per query", micros(ss), true, fmtMicros),
		series: []series{
			{name: "search latency", colour: colOrange, values: micros(ss)},
			{name: "recall@10", colour: colBlue, values: recalls(ss), right: true},
		},
		note: "dim=128 · M=16 · ef=64 · k=10 · recall measured against brute-force ground truth",
	}
	right := recallAxis("recall@10", recalls(ss))
	c.right = &right
	write("latency-vs-corpus-size.svg", c.render())
}

func plotEf(ss []sample) {
	if len(ss) == 0 {
		return
	}
	sortNumericLabels(ss)

	c := &chart{
		title:      "ef: the only knob you can turn per query",
		subtitle:   "Wider search buys recall with latency. It must never find less — that monotonicity is asserted, not hoped for.",
		xLabel:     "ef (search width at query time)",
		categories: labels(ss),
		left:       recallAxis("recall@10", recalls(ss)),
		series: []series{
			{name: "recall@10", colour: colBlue, values: recalls(ss)},
			{name: "search latency", colour: colOrange, values: micros(ss), right: true},
		},
		note: corpusNote(ss) + " · dim=128 · M=16",
	}
	right := niceAxis("µs per query", micros(ss), true, fmtMicros)
	c.right = &right
	write("recall-vs-ef.svg", c.render())
}

func plotM(ss []sample) {
	if len(ss) == 0 {
		return
	}
	sortNumericLabels(ss)

	c := &chart{
		title:      "M: structural, so this is the chart to read before building",
		subtitle:   "More neighbours per node means more ways out of it — paid for once, at build time, and unchangeable after.",
		xLabel:     "M (neighbours per node, layers > 0)",
		categories: labels(ss),
		left:       recallAxis("recall@10", recalls(ss)),
		series: []series{
			{name: "recall@10", colour: colBlue, values: recalls(ss)},
			{name: "build time", colour: colPurple, values: buildSecs(ss), right: true},
		},
		note: corpusNote(ss) + " · dim=128 · ef=64",
	}
	right := niceAxis("build seconds", buildSecs(ss), true, fmtSecs)
	c.right = &right
	write("recall-vs-m.svg", c.render())
}

func plotMetric(ss []sample) {
	if len(ss) == 0 {
		return
	}
	c := &chart{
		title:      "Recall by metric",
		subtitle:   "DotProduct is not a metric — no triangle inequality, and magnitude counts — so a graph is a weaker structure over it.",
		xLabel:     "distance metric",
		categories: labels(ss),
		left:       recallAxis("recall@10", recalls(ss)),
		series: []series{
			{name: "recall@10", colour: colBlue, values: recalls(ss), bar: true},
		},
		note: corpusNote(ss) + " · dim=128 · ef=64",
	}
	write("recall-by-metric.svg", c.render())
}

func plotDistribution(ss []sample) {
	if len(ss) == 0 {
		return
	}
	c := &chart{
		title:      "Uniform noise vs clustered data",
		subtitle:   "Real embeddings cluster, and a graph index exploits exactly that. Uniform random is the pessimistic case, not the typical one.",
		xLabel:     "corpus distribution",
		categories: labels(ss),
		left:       recallAxis("recall@10", recalls(ss)),
		series: []series{
			{name: "recall@10", colour: colGreen, values: recalls(ss), bar: true},
		},
		note: corpusNote(ss) + " · dim=128 · ef=64",
	}
	write("recall-by-distribution.svg", c.render())
}

func plotTombstones(dead, compacted []sample) {
	if len(dead) == 0 || len(compacted) == 0 {
		return
	}
	sortNumericLabels(dead)
	sortNumericLabels(compacted)

	both := append(append([]float64{}, micros(dead)...), micros(compacted)...)
	c := &chart{
		title:      "What tombstones cost, and what Compact gives back",
		subtitle:   "Dead slots ride the frontier and keep results under-filled, so the search widens. Rebuilding returns it to a clean graph.",
		xLabel:     "share of slots tombstoned",
		categories: labels(dead),
		left:       niceAxis("µs per query", both, true, fmtMicros),
		series: []series{
			{name: "with tombstones", colour: colRed, values: micros(dead), bar: true},
			{name: "after Compact()", colour: colGreen, values: micros(compacted), bar: true},
		},
		// Not "recall is unchanged" — the measurements say otherwise, and the
		// reason is the interesting part: dead slots keep results under-filled,
		// which loosens the pruning bound and widens the search beyond what ef
		// asked for. Compaction withdraws that accidental subsidy along with the
		// latency it cost.
		note: "dim=128 · ef=64 · k=10 · at 50% dead: 0.968 recall for 184µs, 0.949 for 88µs once compacted",
	}
	write("tombstones-vs-compaction.svg", c.render())
}
