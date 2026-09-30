// Capability names, checked against the C name table, the mask every call passes, and the devices.
//
// File: golang/capabilities_test.go
// Author: Ash Vardanian

package sz_test

import (
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

func TestCapabilityNames(t *testing.T) {
	for capability, expected := range map[sz.Capability]string{
		sz.CapSerial:      "serial",
		sz.CapWestmere:    "westmere",
		sz.CapGoldmont:    "goldmont",
		sz.CapHaswell:     "haswell",
		sz.CapSkylake:     "skylake",
		sz.CapIcelake:     "icelake",
		sz.CapNeon:        "neon",
		sz.CapNeonAes:     "neonaes",
		sz.CapNeonSha:     "neonsha",
		sz.CapSve:         "sve",
		sz.CapSve2:        "sve2",
		sz.CapSve2Aes:     "sve2aes",
		sz.CapRvv:         "rvv",
		sz.CapRvvCrypto:   "rvvcrypto",
		sz.CapV128:        "v128",
		sz.CapV128Relaxed: "v128relaxed",
		sz.CapLoongsonAsx: "loongsonasx",
		sz.CapPowerVsx:    "powervsx",
		sz.CapCuda:        "cuda",
		sz.CapRocm:        "rocm",
		sz.CapMetal:       "metal",
	} {
		if got := capability.String(); got != expected {
			t.Errorf("Capability(%#x).String(): expected %q, got %q", uint64(capability), expected, got)
		}
	}
}

func TestCapabilitiesEnable(t *testing.T) {
	cpu := sz.CPU()
	enabled, err := cpu.CapabilitiesEnabled()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("StringZilla enabled capabilities: %v", enabled)
	defer cpu.CapabilitiesEnable(enabled)
	detected, _ := cpu.CapabilitiesDetected()
	if enabled != detected&cpu.CapabilitiesCompiled() || !enabled.Has(sz.CapSerial) {
		t.Fatalf("CapabilitiesEnabled() = %v, expected serial plus detected and compiled capabilities", enabled)
	}
	if got, _ := cpu.CapabilitiesEnable(0); got != sz.CapSerial {
		t.Fatalf("CapabilitiesEnable(0) = %v, expected serial alone", got)
	}
	if got, _ := cpu.CapabilitiesEnabled(); got != sz.CapSerial {
		t.Fatalf("CapabilitiesEnabled() after narrowing = %v, expected serial alone", got)
	}
	if got := sz.Index("hello world", "world"); got != 6 {
		t.Errorf("Index on the serial capability: expected 6, got %d", got)
	}
	if got, _ := cpu.CapabilitiesEnable(sz.CapAny); got != enabled {
		t.Errorf("CapabilitiesEnable(CapAny) = %v, expected the original set %v back", got, enabled)
	}
}

func TestDevices(t *testing.T) {
	if sz.CapCpus&sz.CapDevices != 0 || sz.CapCpus|sz.CapDevices|sz.CapAny != sz.CapAny {
		t.Errorf("CapCpus %v and CapDevices %v overlap or escape CapAny", sz.CapCpus, sz.CapDevices)
	}
	if count, err := sz.CountDevices(sz.DeviceCPU); count != 1 || err != nil {
		t.Errorf("CountDevices(DeviceCPU) = %d, %v, expected one CPU", count, err)
	}
	enabled, _ := sz.CPU().CapabilitiesEnabled()
	unlock, err := sz.CPU().ConfigureThread(enabled)
	unlock()
	if err != nil {
		t.Errorf("the CPU refused to configure a thread for its own %v: %v", enabled, err)
	}
	for _, kind := range []sz.DeviceKind{sz.DeviceCPU, sz.DeviceCUDA, sz.DeviceROCm, sz.DeviceMetal} {
		count, _ := sz.CountDevices(kind)
		if _, err := sz.NewDevice(kind, count); err == nil {
			t.Errorf("NewDevice(%d, %d) made a device past the last one", kind, count)
		}
		if count == 0 || kind == sz.DeviceCPU {
			continue
		}
		gpu, err := sz.NewDevice(kind, 0)
		if err != nil || gpu.CapabilitiesCompiled()&sz.CapCpus != 0 {
			t.Errorf("NewDevice(%d, 0) = %v, compiling %v", kind, err, gpu.CapabilitiesCompiled())
		}
		if _, err := gpu.CapabilitiesEnable(sz.CapAny); err == nil {
			t.Errorf("a GPU of kind %d took a CPU enabled set", kind)
		}
		if _, err := gpu.ConfigureThread(sz.CapAny); err == nil {
			t.Errorf("a GPU of kind %d configured a CPU thread", kind)
		}
	}
}
