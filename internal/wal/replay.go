package wal

import "fmt"

// Result reports what a Replay found. It comes back even when Replay returns an
// error, so a caller can see how far recovery got before it stopped.
type Result struct {
	// Records is how many records were handed to the callback.
	Records int

	// Segments is how many segment files were opened.
	Segments int

	// LastSeq is the highest sequence number replayed, or zero for an empty log.
	LastSeq uint64

	// Tears lists the segments that ended in damage. Empty is the normal case —
	// a log closed cleanly has none. One is the normal case after a crash.
	Tears []Tear
}

// NextSeq is the sequence number the log should continue from, ready to hand to
// Options.FirstSeq. On an empty log it is 1, which is what FirstSeq defaults to
// anyway — so a fresh database and a recovered one are opened the same way.
func (r Result) NextSeq() uint64 { return r.LastSeq + 1 }

// Tear is a segment that stopped short: the point where a record failed to
// validate and everything after it in that file was discarded.
type Tear struct {
	// Segment is the index of the segment file.
	Segment uint32

	// Offset is where the damaged record began — the byte the segment is
	// logically truncated at.
	Offset int64

	// Discarded is how many bytes were dropped, from Offset to the end of the
	// file. Worth watching: a torn write leaves a few hundred bytes at most,
	// because the writer flushes in whole buffers. Megabytes means something
	// other than power loss happened.
	Discarded int64

	// Cause is why the record was rejected: ErrShortRecord, ErrChecksum,
	// ErrRecordTooLarge, or ErrInvalidType.
	Cause error
}

func (t Tear) String() string {
	return fmt.Sprintf("%s: %v at offset %d, %d bytes discarded",
		segmentName(t.Segment), t.Cause, t.Offset, t.Discarded)
}

// Replay reads every record in dir, in order, and calls fn for each one.
//
// # Why this is a function and not a method on WAL
//
// Recovery happens *before* a writer exists. Rebuilding state from the log is
// something done to a directory, not to the thing currently appending to it, and
// making it a method would have meant either opening a writer in order to read
// (which creates a segment as a side effect of recovery) or a second constructor
// that returns a WAL that cannot write. Both are worse than a function that
// takes a path.
//
// # Truncation here is logical, not physical
//
// A damaged tail is dropped from the replay; the bytes stay on disk. Nothing
// will ever append to them, because Open always starts a new segment, so
// rewriting the file would buy nothing and cost the one copy of the evidence
// that a crash happened. Recovery is a read.
//
// # The payload is only valid during the callback
//
// fn receives a payload that points into a buffer the reader reuses, which is
// what keeps replay allocation-free per record. A callback that keeps the bytes
// past its return — appending them to a slice, stashing them in a map — must
// copy first, and Record.Clone does exactly that.
//
// A returned error means recovery did not complete and the state fn built is a
// prefix of the log, not the log. The Result is still returned, so the caller
// can report where it stopped.
func Replay(dir string, opts Options, fn func(Record) error) (Result, error) {
	opts = opts.withDefaults()

	// A directory with no segments is not an error — it is a database that has
	// never been written to, which is exactly what a first Open finds.
	indexes, err := listSegments(dir)
	if err != nil {
		return Result{}, fmt.Errorf("wal: list segments: %w", err)
	}

	var res Result
	for _, idx := range indexes {
		if err := replaySegment(dir, idx, opts, fn, &res); err != nil {
			return res, err
		}
	}
	return res, nil
}

// replaySegment scans one segment into res.
//
// # A damaged tail ends this segment, not the replay
//
// Scanning continues with the next segment, and that is load-bearing rather
// than lenient. Open always starts a *new* segment, so the shape a second crash
// leaves behind is: segment K torn, segment K+1 full of perfectly good records
// written after the restart. Refusing to read past a tear in any segment but the
// last would make a database unrecoverable from exactly the situation the writer
// is designed to produce.
//
// It is safe for the same reason it is necessary. A tear can only ever be at the
// end of a segment's written region: the writer's failure is sticky, so it never
// writes past a point it failed at, and rotation fsyncs the old segment before
// the new one is created, so a completed segment is durable before anything
// exists to follow it. Records cannot hide behind a tear, because the writer
// never put any there.
func replaySegment(dir string, idx uint32, opts Options, fn func(Record) error, res *Result) error {
	r, err := openSegmentReader(segmentPath(dir, idx), opts.MaxRecordBytes)
	if err != nil {
		return fmt.Errorf("wal: %s: %w", segmentName(idx), err)
	}
	defer r.close()

	res.Segments++
	for {
		ok, err := r.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}

		// Sequence numbers must strictly increase across the whole log. A gap is
		// fine and expected — it is what a tear leaves behind — but a repeat or a
		// rewind means two records share an identity, and replaying them would
		// apply operations in an order the log does not actually specify. Almost
		// always the cause is a writer opened without carrying FirstSeq forward
		// from a previous recovery.
		if r.rec.Seq <= res.LastSeq {
			return fmt.Errorf("wal: %s: %w: seq %d follows %d",
				segmentName(idx), ErrOutOfOrder, r.rec.Seq, res.LastSeq)
		}
		res.LastSeq = r.rec.Seq
		res.Records++

		if err := fn(r.rec); err != nil {
			return fmt.Errorf("wal: %s: replay seq %d: %w", segmentName(idx), r.rec.Seq, err)
		}
	}

	if r.damage != nil {
		res.Tears = append(res.Tears, Tear{
			Segment:   idx,
			Offset:    r.off,
			Discarded: r.size - r.off,
			Cause:     r.damage,
		})
	}
	return nil
}
