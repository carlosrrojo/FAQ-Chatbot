# src/infrastructure/security/policy_pack.py
from src.domain.ports import IResponsePolicy
from src.domain.models import Classification

class PolicyPack(IResponsePolicy):
    """
    A repository of deterministic, static answers for non-FAQ routes,
    supporting Spanish (es), English (en), French (fr), and German (de).
    """

    def __init__(self) -> None:
        # Templates dictionary structure:
        # { category: { sub_category or "default": { lang: response } } }
        self._templates = {
            "greeting": {
                "default": {
                    "es": "¡Hola! Bienvenido a Espazo Nature. ¿En qué puedo ayudarte hoy?",
                    "en": "Hello! Welcome to Espazo Nature. How can I help you today?",
                    "fr": "Bonjour ! Bienvenue à Espazo Nature. Comment puis-je vous aider aujourd'hui ?",
                    "de": "Hallo! Willkommen bei Espazo Nature. Wie kann ich Ihnen heute helfen?",
                }
            },
            "booking_payment": {
                "default": {
                    "es": "Por motivos de seguridad, no gestionamos reservas ni pagos directamente a través de este chat. Por favor, no compartas los datos de tu tarjeta de crédito ni ninguna otra información de pago por este canal. Visita nuestra web oficial https://espazonature.com/es/ o llámanos al (+34) 981 75 23 27 para formalizar tu reserva de forma segura.",
                    "en": "For security reasons, we do not handle bookings or payments directly via chat. Please do not share your credit card details or any other sensitive payment information in this channel. Visit our official booking website at https://espazonature.com/en/ or call us directly at (+34) 981 75 23 27 to finalize your reservation securely.",
                    "fr": "Pour des raisons de sécurité, nous ne gérons pas les réservations ni les paiements par chat. Veuillez ne pas partager vos coordonnées de carte bancaire ou toute autre information de paiement dans ce canal. Visitez notre site officiel https://espazonature.com/fr/ ou appelez-nous directement au (+34) 981 75 23 27 pour finaliser votre réservation en toute sécurité.",
                    "de": "Aus Sicherheitsgründen wickeln wir Buchungen und Zahlungen nicht direkt per Chat ab. Bitte geben Sie Ihre Kreditkartendaten oder andere sensible Zahlungsinformationen nicht in diesem Kanal an. Besuchen Sie unsere offizielle Buchungswebsite https://espazonature.com/de/ oder rufen Sie uns direkt an, um Ihre Reservierung sicher abzuschließen.",
                }
            },
            "pii_disclosure": {
                "default": {
                    "es": "Gracias por tu mensaje. Por motivos de protección de datos, no necesitamos ni almacenamos datos personales como correos electrónicos, direcciones postales, DNI u otra información identificativa a través de este chat. Si deseas formalizar una reserva o necesitas asistencia personalizada, por favor contacta con nosotros a través de nuestra pagina web https://espazonature.com/es/ o llámanos directamente al (+34) 981 75 23 27.",
                    "en": "Thank you for your message. For data protection reasons, we do not need or store personal data such as email addresses, postal addresses, ID numbers, or other identifying information through this chat. If you wish to make a reservation or need personalised assistance, please contact us through our website https://espazonature.com/en/ or call us directly at (+34) 981 75 23 27.",
                    "fr": "Merci pour votre message. Pour des raisons de protection des données, nous n'avons pas besoin et ne stockons pas de données personnelles telles que les adresses e-mail, les adresses postales, les numéros d'identification ou toute autre information d'identification via ce chat. Si vous souhaitez effectuer une réservation ou avez besoin d'une assistance personnalisée, veuillez nous contacter via https://espazonature.com/fr/ ou appelez-nous directement.",
                    "de": "Vielen Dank für Ihre Nachricht. Aus Datenschutzgründen benötigen und speichern wir keine personenbezogenen Daten wie E-Mail-Adressen, Postanschriften, Ausweisnummern oder andere identifizierende Informationen über diesen Chat. Wenn Sie eine Reservierung vornehmen oder eine persönliche Betreuung benötigen, kontaktieren Sie uns bitte über https://espazonature.com/de/ oder rufen Sie uns direkt an.",
                }
            },
            "privacy_rights": {
                "forget": {
                    "es": "Entendido. Ten en cuenta que este chatbot no tiene acceso a las bases de datos ni puede ejecutar operaciones de borrado directamente. Para ejercer tu derecho al olvido (Art. 17 RGPD) y solicitar la eliminación total de tus datos personales, por favor envía un correo a info@espazonature.com. Procesaremos tu solicitud de inmediato.",
                    "en": "Understood. Please note that this chatbot does not have access to databases and cannot execute deletion operations directly. To exercise your right to be forgotten (Art. 17 GDPR) and request the complete deletion of your personal data, please send an email to info@espazonature.com. We will process your request promptly.",
                    "fr": "Compris. Veuillez noter que ce chatbot n'a pas accès aux bases de données et ne peut pas exécuter directement d'opérations de suppression. Pour exercer votre droit à l'oubli (art. 17 du RGPD) et demander la suppression complète de vos données personnelles, veuillez envoyer un e-mail à info@espazonature.com. Nous traiterons votre demande rapidement.",
                    "de": "Verstanden. Bitte beachten Sie, dass dieser Chatbot keinen Zugriff auf Datenbanken hat und Löschvorgänge nicht direkt ausführen kann. Um Ihr Recht auf Vergessenwerden (Art. 17 DSGVO) auszuüben und die vollständige Löschung Ihrer personenbezogenen Daten zu beantragen, senden Sie bitte eine E-Mail an info@espazonature.com. Wir werden Ihre Anfrage umgehend bearbeiten.",
                },
                "access": {
                    "es": "Ten en cuenta que este chatbot no tiene acceso a las bases de datos ni puede recuperar tu información personal directamente. Para solicitar acceso a tus datos personales o un historial de tus interacciones (Art. 15 RGPD), puedes enviar tu solicitud formal a nuestro delegado de protección de datos en info@espazonature.com.",
                    "en": "Please note that this chatbot does not have access to databases and cannot retrieve your personal information directly. To request access to your personal data or a history of your interactions (Art. 15 GDPR), please send a formal request to our data protection officer at info@espazonature.com.",
                    "fr": "Veuillez noter que ce chatbot n'a pas accès aux bases de données et ne peut pas récupérer directement vos informations personnelles. Pour demander l'accès à vos données personnelles ou l'historique de vos interactions (art. 15 du RGPD), veuillez envoyer une demande formelle à notre délégué à la protection des données à l'adresse info@espazonature.com.",
                    "de": "Bitte beachten Sie, dass dieser Chatbot keinen Zugriff auf Datenbanken hat und Ihre personenbezogenen Daten nicht direkt abrufen kann. Um Auskunft über Ihre personenbezogenen Daten oder einen Verlauf Ihrer Interaktionen zu beantragen (Art. 15 DSGVO), senden Sie bitte eine formelle Anfrage an unseren Datenschutzbeauftragten unter info@espazonature.com.",
                },
                "retention": {
                    "es": "De acuerdo con nuestra política de privacidad, las sesiones de chat y sus datos asociados se eliminan de forma segura transcurrido un periodo máximo de 30 días.",
                    "en": "In accordance with our privacy policy, chat sessions and associated data are securely deleted after a maximum period of 30 days.",
                    "fr": "Conformément à notre politique de confidentialité, les sessions de chat et leurs données associées sont supprimées en toute sécurité après une période maximale de 30 jours.",
                    "de": "In Übereinstimmung mit unserer Datenschutzerklärung werden Chat-Sitzungen und die damit verbundenen Daten nach Ablauf einer maximalen Frist von 30 Tagen sicher gelöscht.",
                },
                "minors": {
                    "es": "Para informarte sobre nuestros alojamientos y servicios, no necesitamos documentación ni datos personales de menores de edad. Este chatbot no recopila ni procesa datos de menores. Para formalizar el registro de huéspedes menores, por favor contacta con nosotros a través de nuestra pagina web https://espazonature.com/es/ o llamando al +34 981 75 23 27.",
                    "en": "To provide you with information about our accommodation and services, we do not need documentation or personal data of minors. This chatbot does not collect or process minors' data. To formalize the registration of minor guests, please contact us through our website https://espazonature.com/en/ or by calling +34 981 75 23 27.",
                    "fr": "Pour vous informer sur nos hébergements et services, nous n'avons pas besoin de documentation ni de données personnelles de mineurs. Ce chatbot ne collecte ni ne traite les données de mineurs. Pour formaliser l'enregistrement de clients mineurs, veuillez nous contacter via notre site web https://espazonature.com/fr/ ou en appelant le +34 981 75 23 27.",
                    "de": "Um Ihnen Informationen über unsere Unterkünfte und Dienstleistungen zu geben, benötigen wir keine Dokumente oder personenbezogene Daten von Minderjährigen. Dieser Chatbot erfasst oder verarbeitet keine Daten von Minderjährigen. Für die Registrierung minderjähriger Gäste wenden Sie sich bitte über unsere Website https://espazonature.com/de/ oder telefonisch unter +34 981 75 23 27 an uns.",
                },
                "third_party": {
                    "es": "Para proteger la privacidad de terceros, no recopilamos ni procesamos nombres, DNI u otros datos de acompañantes o terceras personas a través de este chat. Todo el registro de huéspedes adicionales debe realizarse de forma segura durante el check-in o a través de nuestra pagina web https://espazonature.com/es/.",
                    "en": "To protect the privacy of third parties, we do not collect or process names, DNI, or other data of companions or third parties through this chat. All additional guest registration must be completed securely during check-in or through our page web https://espazonature.com/en/.",
                    "fr": "Pour protéger la vie privée des tiers, nous ne collectons ni ne traitons les noms, DNI ou autres données de compagnons ou de tiers via ce chat. Tout enregistrement de client supplémentaire doit être effectué en toute sécurité lors de l'enregistrement ou via notre page web https://espazonature.com/fr/.",
                    "de": "Um die Privatsphäre Dritter zu schützen, erfassen oder verarbeiten wir keine Namen, DNI oder andere Daten von Begleitpersonen oder Dritten über diesen Chat. Die Registrierung zusätzlicher Gäste muss sicher beim Check-in oder über unsere Seite web https://espazonature.com/de/ erfolgen.",
                },
                "general_policy": {
                    "es": "Cumplimos estrictamente con el Reglamento General de Protección de Datos (RGPD). Toda la información se almacena de forma segura en territorio de la UE. Para consultar la política completa, visita espazonature.es/privacidad.",
                    "en": "We strictly comply with the General Data Protection Regulation (GDPR). All information is stored securely within the EU. To view our full privacy policy, please visit espazonature.es/privacidad.",
                    "fr": "Nous respectons strictement le Règlement général sur la protection des données (RGPD). Toutes les informations sont stockées en toute sécurité sur le territoire de l'UE. Pour consulter l'intégralité de notre politique, veuillez visiter espazonature.es/privacidad.",
                    "de": "Wir halten uns streng an die Datenschutz-Grundverordnung (DSGVO). Alle Informationen werden sicher innerhalb der EU gespeichert. Die vollständige Datenschutzerklärung finden Sie unter espazonature.es/privacidad.",
                },
                "default": {
                    "es": "Cumplimos estrictamente con el Reglamento General de Protección de Datos (RGPD). Toda la información se almacena de forma segura en territorio de la UE. Para consultar la política completa, visita espazonature.es/privacidad.",
                    "en": "We strictly comply with the General Data Protection Regulation (GDPR). All information is stored securely within the EU. To view our full privacy policy, please visit espazonature.es/privacidad.",
                    "fr": "Nous respectons strictement le Règlement général sur la protection des données (RGPD). Toutes les informations sont stockées en toute sécurité sur le territoire de l'UE. Pour consulter l'intégralité de notre politique, veuillez visiter espazonature.es/privacidad.",
                    "de": "Wir halten uns streng an die Datenschutz-Grundverordnung (DSGVO). Alle Informationen werden sicher innerhalb der EU gespeichert. Die vollständige Datenschutzerklärung finden Sie unter espazonature.es/privacidad.",
                }
            },
            "injection": {
                "default": {
                    "es": "Lo siento, no puedo procesar esta solicitud. Solo dispongo de la información publicada oficialmente por Espazo Nature. Si tienes alguna otra pregunta sobre los alojamientos, actividades o servicios, estaré encantado de ayudarte.",
                    "en": "I'm sorry, I cannot process this request. I only have access to officially published information from Espazo Nature. If you have any questions about the accommodation, activities, or services, I would be happy to help.",
                    "fr": "Désolé, je ne peux pas traiter cette demande. Je ne dispose que des informations officiellement publiées par Espazo Nature. Si vous avez des questions sur l'hébergement, les activités ou les services, je me ferai un plaisir de vous aider.",
                    "de": "Es tut mir leid, ich kann diese Anfrage nicht verarbeiten. Ich verfüge nur über die offiziell von Espazo Nature veröffentlichten Informationen. Wenn Sie Fragen zu den Unterkünften, Aktivitäten oder Dienstleistungen haben, helfe ich Ihnen gerne weiter.",
                }
            },
            "unsupported": {
                "default": {
                    "es": "Actualmente solo puedo procesar mensajes de texto. Por favor, envía tus dudas en formato de texto y estaré encantado de ayudarte.",
                    "en": "Currently, I can only process text messages. Please send your queries in text format, and I will be happy to help.",
                    "fr": "Actuellement, je ne peux traiter que les messages textuels. Veuillez envoyer vos questions au format texte et je me ferai un plaisir de vous aider.",
                    "de": "Derzeit kann ich nur Textnachrichten verarbeiten. Bitte senden Sie Ihre Fragen im Textformat und ich helfe Ihnen gerne weiter.",
                }
            }
        }

    def get_response(self, classification: Classification) -> str:
        """
        Look up the static response for a given classification.
        Falls back to 'es' if language not supported, and to 'default' sub_category if sub_category not found.
        """
        category = classification.category.value if hasattr(classification.category, "value") else str(classification.category)
        sub_category = classification.sub_category or "none"
        language = classification.language or "es"

        # Validate/sanitize language
        lang = language.lower() if language else "es"
        if lang not in ["es", "en", "fr", "de"]:
            lang = "es"

        # Lookup category
        cat_dict = self._templates.get(category)
        if not cat_dict:
            # Fallback for unknown category
            return self._templates["unsupported"]["default"][lang]

        # Lookup subcategory or default
        sub = sub_category.lower() if sub_category else "none"
        sub_dict = cat_dict.get(sub)
        if not sub_dict:
            sub_dict = cat_dict.get("default")
        
        if not sub_dict:
            # Fallback
            sub_dict = self._templates["unsupported"]["default"]

        return sub_dict.get(lang, sub_dict.get("es", ""))
