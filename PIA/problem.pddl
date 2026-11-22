
(define (problem incidente-servidor-web)

    (:domain respuesta-incidentes)

    (:objects
        servidor_web servidor_db servidor_archivos - servidor
    )

    (:init
        (servidor-comprometido servidor_web)

        (servicio-activo servidor_web)
        (servicio-activo servidor_db)
        (servicio-activo servidor_archivos)
    )

    (:goal
        (and
            (servidor-verificado servidor_web)
            (vulnerabilidad-corregida servidor_web)
            (servicio-activo servidor_web)
        )
    )
)
