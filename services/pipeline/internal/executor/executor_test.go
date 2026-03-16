package executor

import (
	"testing"

	"kk-infra/services/pipeline"
)

// 完整流水线：5 阶段执行
func TestRun(t *testing.T) {
	e := NewDefaultExecutor()
	rec, err := e.Run("v-test-1", "admin")
	if err != nil {
		t.Fatalf("执行失败: %v", err)
	}
	if rec.ModelVersionID != "v-test-1" {
		t.Errorf("ModelVersionID 错误: %s", rec.ModelVersionID)
	}
	if len(rec.StageResults) != 5 {
		t.Fatalf("应有 5 个阶段，实际 %d", len(rec.StageResults))
	}
	// 阶段顺序：artifact → probe → benchmark → approval → release
	expected := []pipeline.Stage{
		pipeline.StageArtifactValidate,
		pipeline.StageProbe,
		pipeline.StageBenchmark,
		pipeline.StageApproval,
		pipeline.StageRelease,
	}
	for i, want := range expected {
		if rec.StageResults[i].Stage != want {
			t.Errorf("阶段 %d = %s, want %s", i, rec.StageResults[i].Stage, want)
		}
	}
	// 基准测试默认 skipped
	if rec.StageResults[2].Status != "skipped" {
		t.Errorf("默认应跳过基准测试: %s", rec.StageResults[2].Status)
	}
}

// 空版本 ID 报错
func TestRunEmptyVersion(t *testing.T) {
	e := NewDefaultExecutor()
	if _, err := e.Run("", "admin"); err == nil {
		t.Fatal("空版本 ID 应报错")
	}
}

// 基准测试开启时跑 benchmark
func TestRunBenchmark(t *testing.T) {
	e := NewDefaultExecutor()
	e.SkipBenchmark = false
	rec, err := e.Run("v-test-2", "admin")
	if err != nil {
		t.Fatalf("执行失败: %v", err)
	}
	for _, s := range rec.StageResults {
		if s.Stage == pipeline.StageBenchmark && s.Status != "passed" {
			t.Errorf("基准测试应通过: %s", s.Status)
		}
	}
	bench, err := e.RunBenchmark("v-test-2")
	if err != nil {
		t.Fatalf("基准测试失败: %v", err)
	}
	if bench.TTFTMs <= 0 || bench.TokensPerSec <= 0 {
		t.Errorf("基准测试结果异常: %+v", bench)
	}
}
