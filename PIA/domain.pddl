
(define (domain respuesta-incidentes)

    (:requirements
        :strips
        :typing
    )

    (:types
        servidor
    )

    (:predicates
        (servidor-comprometido ?s - servidor)
        (servidor-aislado ?s - servidor)
        (evidencia-recolectada ?s - servidor)
        (vulnerabilidad-identificada ?s - servidor)
        (vulnerabilidad-corregida ?s - servidor)
        (servidor-verificado ?s - servidor)
        (servicio-activo ?s - servidor)
        (incidente-contenido ?s - servidor)
    )

    (:action aislar-servidor
        :parameters (?s - servidor)
        :precondition
            (and
                (servidor-comprometido ?s)
                (servicio-activo ?s)
            )
        :effect
            (and
                (servidor-aislado ?s)
                (incidente-contenido ?s)
                (not (servicio-activo ?s))
            )
    )

    (:action recolectar-evidencia
        :parameters (?s - servidor)
        :precondition
            (and
                (servidor-comprometido ?s)
                (servidor-aislado ?s)
                (incidente-contenido ?s)
            )
        :effect
            (evidencia-recolectada ?s)
    )

    (:action identificar-vulnerabilidad
        :parameters (?s - servidor)
        :precondition
            (evidencia-recolectada ?s)
        :effect
            (vulnerabilidad-identificada ?s)
    )

    (:action corregir-vulnerabilidad
        :parameters (?s - servidor)
        :precondition
            (and
                (servidor-aislado ?s)
                (vulnerabilidad-identificada ?s)
            )
        :effect
            (and
                (vulnerabilidad-corregida ?s)
                (not (servidor-comprometido ?s))
            )
    )

    (:action verificar-servidor
        :parameters (?s - servidor)
        :precondition
            (and
                (servidor-aislado ?s)
                (vulnerabilidad-corregida ?s)
            )
        :effect
            (servidor-verificado ?s)
    )

    (:action restaurar-servicio
        :parameters (?s - servidor)
        :precondition
            (and
                (servidor-aislado ?s)
                (servidor-verificado ?s)
                (vulnerabilidad-corregida ?s)
            )
        :effect
            (and
                (servicio-activo ?s)
                (not (servidor-aislado ?s))
            )
    )
)
