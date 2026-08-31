"""Ders ve uygulama düzeyindeki sabit yapılandırma."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CourseConfig:
    course_code: str
    course_name: str
    application_subtitle: str
    program_name: str
    institution_name: str
    academic_year: str


APP_CONFIG = CourseConfig(
    course_code="IKT 807",
    course_name="Ekonometrik Modelleme ve Uygulamaları",
    application_subtitle="Etkileşimli Ekonometrik Araştırma Laboratuvarı",
    program_name="İktisat Tezli Yüksek Lisans Programı",
    institution_name="İzmir Bakırçay Üniversitesi",
    academic_year="2026-2027 Eğitim-Öğretim Yılı",
)
